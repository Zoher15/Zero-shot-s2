"""
OpenAI O3 Batch Processing Module

Functional data-driven architecture for OpenAI O3 batch API requests with group-based 
submission, validation, status tracking, and DeepSeek post-processing.
"""

import os
import time
import json
import tempfile
import shutil
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Callable, Any, NamedTuple, Optional
from functools import partial
import config

# Setup logging
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION AND DATA STRUCTURES
# =============================================================================

@dataclass
class BatchConfig:
    """Configuration for batch processing pipeline."""
    chunk_size: int = 500
    max_size_mb: int = 200
    completion_window: str = "24h"
    retry_attempts: int = 3
    status_check_interval: int = 60
    json_indent: int = 4
    terminal_statuses: set = None
    
    def __post_init__(self):
        if self.terminal_statuses is None:
            self.terminal_statuses = {'completed', 'failed', 'expired', 'cancelled'}

@dataclass
class PipelineStep:
    """Represents a single step in the processing pipeline."""
    name: str
    function: Callable
    retry: bool = False
    required: bool = True
    description: str = ""

@dataclass
class MinimalBatchInfo:
    """Minimal batch tracking info - just what we need."""
    batch_id: str
    chunk_id: int
    chunk_start_idx: int
    chunk_end_idx: int
    chunk_size: int

class BatchData(NamedTuple):
    """Immutable data container for batch processing state."""
    examples: tuple
    chunks: tuple = ()
    group_dir: Optional[Path] = None
    tracker: dict = {}
    responses: dict = {}
    metadata: dict = {}
    config: Optional[BatchConfig] = None
    
    def with_chunks(self, chunks):
        return self._replace(chunks=tuple(chunks))
    
    def with_group_dir(self, group_dir):
        return self._replace(group_dir=group_dir)
    
    def with_tracker(self, tracker):
        return self._replace(tracker=tracker)
    
    def with_responses(self, responses):
        return self._replace(responses=responses)
    
    def with_metadata(self, **metadata):
        new_metadata = {**self.metadata, **metadata}
        return self._replace(metadata=new_metadata)

# =============================================================================
# PIPELINE EXECUTION ENGINE
# =============================================================================

def execute_pipeline(data: BatchData, steps: List[PipelineStep]) -> BatchData:
    """Execute a pipeline of steps on batch data."""
    current_data = data
    
    for step in steps:
        logger.info(f"Executing step: {step.name}")
        try:
            if step.retry:
                step_func = with_retry(current_data.config.retry_attempts)(step.function)
            else:
                step_func = step.function
                
            current_data = step_func(current_data)
            logger.info(f"✅ Completed step: {step.name}")
            
        except Exception as e:
            if step.required:
                logger.error(f"❌ Required step failed: {step.name} - {e}")
                raise
            else:
                logger.warning(f"⚠️ Optional step failed: {step.name} - {e}")
                
    return current_data

# =============================================================================
# PURE UTILITY FUNCTIONS
# =============================================================================

def get_openai_clients():
    """Get OpenAI clients - isolated I/O dependency."""
    from openai import OpenAI
    
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    deepseek_client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/beta"
    )
    return openai_client, deepseek_client

def with_retry(max_attempts: int = 3, delay_base: float = 2.0):
    """Higher-order function for retry logic."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    delay = delay_base ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

def with_logging(operation_name: str):
    """Higher-order function for operation logging."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            logger.info(f"Starting {operation_name}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"✅ Completed {operation_name}")
                return result
            except Exception as e:
                logger.error(f"❌ Failed {operation_name}: {e}")
                raise
        return wrapper
    return decorator

def safe_file_operation(file_path: Path, operation: str, data: Any = None, encoding: str = 'utf-8'):
    """Unified file operations with error handling."""
    if operation == "read_json":
        with open(file_path, 'r', encoding=encoding) as f:
            return json.load(f)
    elif operation == "write_json":
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding=encoding) as f:
            json.dump(data, f, indent=4)
        return file_path
    elif operation == "read_lines":
        with open(file_path, 'r', encoding=encoding) as f:
            return [line.strip() for line in f]
    else:
        raise ValueError(f"Unknown file operation: {operation}")

def create_temp_jsonl_file(requests: List[dict]) -> str:
    """Create temporary JSONL file and return path."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for req in requests:
            json.dump(req, f)
            f.write('\n')
        return f.name

def measure_file_size_mb(file_path: str) -> float:
    """Get file size in MB."""
    return os.path.getsize(file_path) / (1024 * 1024)

def cleanup_temp_file(file_path: str):
    """Safely remove temporary file."""
    try:
        os.unlink(file_path)
    except (OSError, FileNotFoundError):
        pass

# =============================================================================
# PIPELINE STEP FUNCTIONS
# =============================================================================

def create_chunks_pure(data: BatchData) -> BatchData:
    """Create chunks from examples based on config."""
    chunk_size = data.config.chunk_size
    examples = data.examples
    
    chunks = []
    for i in range(0, len(examples), chunk_size):
        chunk_end = min(i + chunk_size, len(examples))
        chunk_examples = examples[i:chunk_end]
        chunk_id = i // chunk_size + 1
        
        chunks.append({
            'chunk_id': chunk_id,
            'chunk_idx': i,
            'chunk_end': chunk_end,
            'chunk_examples': chunk_examples,
            'status': 'pending'
        })
    
    return data.with_chunks(chunks)

def setup_group_directory_pure(data: BatchData) -> BatchData:
    """Setup group directory with datetime naming, checking for existing directories first."""
    dataset_name = data.metadata.get('dataset_name', 'images')
    mode_type = data.metadata.get('mode_type', 'zeroshot')
    model_name = data.metadata.get('model_name', 'o3')
    num_examples = len(data.examples)
    
    batches_dir = Path(config.OUTPUTS_DIR) / 'batches'
    
    # Look for existing directories matching this configuration
    if batches_dir.exists():
        pattern = f"{model_name}_group_{dataset_name}_{mode_type}_{num_examples}_*"
        existing_dirs = list(batches_dir.glob(pattern))
        logger.info(f"Searching for pattern: {pattern}")
        logger.info(f"Found {len(existing_dirs)} matching directories: {[d.name for d in existing_dirs]}")
        
        if existing_dirs:
            # Sort by creation time (newest first) and use the most recent
            existing_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            group_dir = existing_dirs[0]
            
            # Extract datetime from existing directory name
            dir_name = group_dir.name
            parts = dir_name.split('_')
            timestamp_part = f"{parts[-2]}_{parts[-1]}"  # Get last two parts: YYYYMMDD_HHMMSS
            try:
                current_datetime = datetime.strptime(timestamp_part, "%Y%m%d_%H%M%S")
                group_name = timestamp_part
                logger.info(f"Found existing group directory: {group_dir}")
            except ValueError:
                # Fallback if timestamp parsing fails
                current_datetime = datetime.fromtimestamp(group_dir.stat().st_mtime)
                group_name = timestamp_part
            
            return data.with_group_dir(group_dir).with_metadata(
                group_name=group_name,
                current_datetime=current_datetime,
                found_existing=True
            )
    
    # No existing directory found, create new one
    current_datetime = datetime.now()
    group_name = current_datetime.strftime("%Y%m%d_%H%M%S")
    group_dir = batches_dir / f"{model_name}_group_{dataset_name}_{mode_type}_{num_examples}_{group_name}"
    
    return data.with_group_dir(group_dir).with_metadata(
        group_name=group_name,
        current_datetime=current_datetime,
        found_existing=False
    )

def handle_existing_group_pure(data: BatchData) -> BatchData:
    """Handle existing group detection and user decision."""
    if not data.group_dir.exists():
        return data.with_metadata(overwrite_decision=None, resume_existing=False)
    
    readable_datetime = data.metadata['current_datetime'].strftime("%B %d, %Y at %H:%M:%S")
    print(f"Found existing O3 group from {readable_datetime}.")
    
    try:
        user_input = input("Do you want to overwrite this group and its submitted chunks? (y/N): ").strip().lower()
        overwrite_decision = user_input == 'y'
    except EOFError:
        # Non-interactive mode - default to resume
        logger.info("Non-interactive mode - automatically resuming existing group")
        overwrite_decision = False
    
    if overwrite_decision:
        shutil.rmtree(data.group_dir)
        logger.info(f"Overwriting existing group: {data.metadata['group_name']}")
    else:
        logger.info(f"Resuming existing group: {data.metadata['group_name']}")
    
    return data.with_metadata(
        overwrite_decision=overwrite_decision,
        resume_existing=not overwrite_decision
    )

def create_group_directory_pure(data: BatchData) -> BatchData:
    """Create group directory if needed."""
    overwrite = data.metadata.get('overwrite_decision')
    
    if overwrite or not data.group_dir.exists():
        data.group_dir.mkdir(parents=True, exist_ok=True)
        readable_datetime = data.metadata['current_datetime'].strftime("%B %d, %Y at %H:%M:%S")
        logger.info(f"Created {data.metadata.get('model_name', 'OpenAI')} group directory: {readable_datetime} ({data.metadata['group_name']})")
    
    return data

def validate_chunk_sizes_pure(data: BatchData) -> BatchData:
    """Validate all chunk sizes in parallel before submission."""
    import concurrent.futures
    
    model_kwargs = data.metadata.get('model_kwargs', {})
    
    def validate_single_chunk(chunk_info):
        """Validate a single chunk size."""
        chunk_id = chunk_info['chunk_id']
        chunk_examples = chunk_info['chunk_examples']
        
        # Create batch requests for validation
        batch_requests, _ = create_batch_requests_for_chunk_pure(chunk_id, chunk_examples, model_kwargs, data.metadata.get('model_name', 'o3'))
        
        # Measure size
        temp_file = create_temp_jsonl_file(batch_requests)
        try:
            file_size_mb = measure_file_size_mb(temp_file)
            return chunk_id, file_size_mb, file_size_mb > data.config.max_size_mb
        finally:
            cleanup_temp_file(temp_file)
    
    logger.info(f"Validating {len(data.chunks)} chunk sizes in parallel...")
    oversized_chunks = []
    
    # Validate all chunks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(data.chunks))) as executor:
        future_to_chunk = {
            executor.submit(validate_single_chunk, chunk_info): chunk_info
            for chunk_info in data.chunks
        }
        
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_id, file_size_mb, is_oversized = future.result()
            
            if is_oversized:
                oversized_chunks.append((chunk_id, file_size_mb))
                logger.warning(f"Chunk {chunk_id} is {file_size_mb:.1f}MB (exceeds {data.config.max_size_mb}MB limit)")
            else:
                logger.debug(f"Chunk {chunk_id}: {file_size_mb:.1f}MB ✓")
    
    if oversized_chunks:
        if data.group_dir.exists():
            shutil.rmtree(data.group_dir)
        
        oversized_info = ", ".join([f"Chunk {chunk_id}: {size_mb:.1f}MB" for chunk_id, size_mb in oversized_chunks])
        error_msg = f"""
❌ OpenAI Batch Size Validation Failed

The following chunks exceed the {data.config.max_size_mb}MB limit:
{oversized_info}

Current chunk size: {data.config.chunk_size} examples per chunk
Total chunks created: {len(data.chunks)}

Please reduce the chunk_size parameter and try again.
Recommended: Try chunk_size = {max(100, data.config.chunk_size // 2)}
"""
        print(error_msg)
        raise ValueError(f"Chunks exceed {data.config.max_size_mb}MB limit. Reduce chunk_size from {data.config.chunk_size}.")
    
    logger.info(f"✅ All {len(data.chunks)} chunks are under {data.config.max_size_mb}MB limit")
    return data

def create_batch_requests_for_chunk_pure(chunk_id, chunk_examples, model_kwargs, model_name):
    """Pure function to create batch requests for a chunk."""
    batch_requests = []
    example_metadata = []
    timestamp = int(time.time())
    
    for i, example in enumerate(chunk_examples):
        request_id = f"req_{chunk_id}_{timestamp}_{i:06d}"
        
        if model_name == 'o3':
            # O3 uses reasoning API format
            request_body = {
                "model": "o3",
                "input": example['messages'],
                "reasoning": {"effort": "high", "summary": "detailed"}
            }
        elif model_name == 'gpt-4.1':
            # GPT-4.1 also uses reasoning API format
            request_body = {
                "model": "gpt-4.1-2025-04-14",
                "input": example['messages']
            }
        else:
            raise ValueError(f"Unsupported OpenAI model: {model_name}")
        
        # Both models use the same endpoint
        endpoint = "/v1/responses"
        
        # Add model-specific parameters
        if 'temperature' in model_kwargs and model_name == 'o3':
            request_body['temperature'] = model_kwargs['temperature']
        if 'top_p' in model_kwargs and model_name == 'o3':
            request_body['top_p'] = model_kwargs['top_p']
        if 'max_tokens' in model_kwargs and model_name == 'gpt-4.1':
            request_body['max_tokens'] = model_kwargs['max_tokens']
        
        batch_requests.append({
            "custom_id": request_id,
            "method": "POST", 
            "url": endpoint,
            "body": request_body
        })
        
        example_metadata.append({
            'request_id': request_id,
            'example_index': i,
            'example': example
        })
    
    return batch_requests, example_metadata

def submit_chunks_pure(data: BatchData) -> BatchData:
    """Submit all validated chunks in parallel."""
    import concurrent.futures
    
    openai_client, _ = get_openai_clients()
    model_kwargs = data.metadata.get('model_kwargs', {})
    
    def submit_single_chunk(chunk_info):
        """Submit a single chunk to OpenAI."""
        chunk_id = chunk_info['chunk_id']
        chunk_examples = chunk_info['chunk_examples']
        
        # Create and submit batch
        batch_requests, example_metadata = create_batch_requests_for_chunk_pure(chunk_id, chunk_examples, model_kwargs, data.metadata.get('model_name', 'o3'))
        temp_file = create_temp_jsonl_file(batch_requests)
        
        try:
            with open(temp_file, 'rb') as f:
                file_id = openai_client.files.create(file=f, purpose='batch').id
            
            # Determine endpoint based on model
            model_name = data.metadata.get('model_name', 'o3')
            endpoint = "/v1/responses"  # Both O3 and GPT-4.1 use reasoning API
            
            batch_response = openai_client.batches.create(
                input_file_id=file_id,
                endpoint=endpoint,
                completion_window=data.config.completion_window
            )
            
            batch_id = batch_response.id
            logger.info(f"Chunk {chunk_id} submitted with batch_id: {batch_id}")
            
            return chunk_id, {
                'batch_id': batch_id,
                'status': 'submitted',
                'chunk_size': len(chunk_examples),
                'chunk_start_idx': chunk_info['chunk_idx'],
                'chunk_end_idx': chunk_info['chunk_end'],
                'submission_time': time.time()
            }
            
        finally:
            cleanup_temp_file(temp_file)
    
    logger.info(f"Submitting {len(data.chunks)} chunks in parallel...")
    
    group_tracker = {
        'group_name': data.group_dir.name,
        'dataset_name': data.metadata.get('dataset_name', 'images'),
        'mode_type': data.metadata.get('mode_type', 'zeroshot'),
        'total_examples': len(data.examples),
        'total_chunks': len(data.chunks),
        'submission_time': time.time(),
        'model_kwargs': model_kwargs,
        'chunks': {}
    }
    
    # Submit all chunks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(data.chunks))) as executor:
        future_to_chunk = {
            executor.submit(submit_single_chunk, chunk_info): chunk_info
            for chunk_info in data.chunks
        }
        
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_id, chunk_data = future.result()
            group_tracker['chunks'][chunk_id] = chunk_data
    
    # Save minimal tracker
    batch_infos = []
    for chunk_id, chunk_data in group_tracker['chunks'].items():
        batch_infos.append({
            'batch_id': chunk_data['batch_id'],
            'chunk_id': int(chunk_id),
            'chunk_start_idx': chunk_data['chunk_start_idx'],
            'chunk_end_idx': chunk_data['chunk_end_idx'],
            'chunk_size': chunk_data['chunk_size']
        })
    
    metadata = {
        'group_name': group_tracker['group_name'],
        'dataset_name': group_tracker['dataset_name'],
        'mode_type': group_tracker['mode_type'],
        'total_examples': group_tracker['total_examples'],
        'total_chunks': group_tracker['total_chunks'],
        'submission_time': group_tracker['submission_time'],
        'model_kwargs': group_tracker['model_kwargs']
    }
    
    # Write minimal tracker files
    batch_file = data.group_dir / 'batches.jsonl'
    with open(batch_file, 'w') as f:
        for info in batch_infos:
            json.dump(info, f)
            f.write('\n')
    
    meta_file = data.group_dir / 'metadata.json'
    safe_file_operation(meta_file, 'write_json', metadata)
    
    logger.info(f"Minimal tracker saved: {batch_file}, {meta_file}")
    
    return data.with_tracker(group_tracker)

def wait_for_completion_pure(data: BatchData) -> BatchData:
    """Wait for all chunks to complete using OpenAI API directly (no heavy file writes)."""
    openai_client, _ = get_openai_clients()
    
    # Use the tracker from previous step (already contains batch info)
    tracker = data.tracker
    
    while True:
        all_completed = True
        status_summary = {}
        
        # Check status of each chunk sequentially via API
        for chunk_id, chunk_info in tracker['chunks'].items():
            batch_id = chunk_info['batch_id']
            
            try:
                status_info = openai_client.batches.retrieve(batch_id)
                current_status = status_info.status
                
                # Update tracker in memory (no file writes)
                tracker['chunks'][chunk_id]['status'] = current_status
                if current_status == 'completed':
                    tracker['chunks'][chunk_id]['completion_time'] = time.time()
                    tracker['chunks'][chunk_id]['output_file_id'] = status_info.output_file_id
                elif current_status in data.config.terminal_statuses:
                    if current_status != 'completed':
                        tracker['chunks'][chunk_id]['error'] = f"Batch failed: {current_status}"
                        logger.error(f"Chunk {chunk_id} failed: {current_status}")
                
                status_summary[current_status] = status_summary.get(current_status, 0) + 1
                
                if current_status not in data.config.terminal_statuses:
                    all_completed = False
                    
            except Exception as e:
                logger.error(f"Error checking chunk {chunk_id}: {e}")
                all_completed = False
        
        # Print status summary (no file writes)
        status_str = ", ".join([f"{status}: {count}" for status, count in status_summary.items()])
        print(f"Group status - {status_str}")
        
        if all_completed:
            break
            
        time.sleep(data.config.status_check_interval)
    
    logger.info("All chunks in group have completed")
    return data.with_tracker(tracker)

def process_with_deepseek_pure(data: BatchData) -> BatchData:
    """Process all completed chunks through DeepSeek in parallel."""
    import concurrent.futures
    
    openai_client, deepseek_client = get_openai_clients()
    
    # Step 1: Download and parse ALL chunk results in parallel
    def download_and_parse_chunk(chunk_data):
        """Download and parse O3 results for a single chunk."""
        chunk_id, chunk_info = chunk_data
        
        if chunk_info['status'] != 'completed':
            logger.warning(f"Skipping chunk {chunk_id} - status: {chunk_info['status']}")
            return []
        
        # Download O3 results for this chunk
        output_file_id = chunk_info['output_file_id']
        
        with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as f:
            f.write(openai_client.files.content(output_file_id).content)
            result_file = f.name
        
        try:
            chunk_requests = []
            
            # Parse O3 batch results without needing stored example_metadata
            with open(result_file, 'r', encoding='utf-8') as f:
                for line in f:
                    result = json.loads(line.strip())
                    custom_id = result['custom_id']
                    
                    # Extract example index from custom_id format: "req_{chunk_id}_{timestamp}_{example_index:06d}"
                    try:
                        parts = custom_id.split('_')
                        example_index = int(parts[-1])  # Last part is the zero-padded example index
                    except (ValueError, IndexError):
                        logger.warning(f"Could not parse custom_id: {custom_id}")
                        continue
                    
                    if result.get('error'):
                        logger.warning(f"Request error in chunk {chunk_id}: {result['error']}")
                        chunk_requests.append((int(chunk_id), example_index, ''))
                        continue
                        
                    response_body = result['response']['body']
                    model_name = data.metadata.get('model_name', 'o3')
                    intermediate_response = parse_openai_response_pure(response_body, model_name)
                    
                    chunk_requests.append((int(chunk_id), example_index, intermediate_response))
            
            return chunk_requests
                
        finally:
            cleanup_temp_file(result_file)
    
    logger.info("Downloading and parsing all chunk results in parallel...")
    all_requests = []
    
    # Download and parse all chunks in parallel
    chunk_items = list(data.tracker['chunks'].items())
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(chunk_items))) as executor:
        future_to_chunk = {
            executor.submit(download_and_parse_chunk, chunk_data): chunk_data
            for chunk_data in chunk_items
        }
        
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_requests = future.result()
            all_requests.extend(chunk_requests)
    
    logger.info(f"Collected {len(all_requests)} intermediate responses from all chunks")
    
    # Step 2: Process ALL requests through DeepSeek in parallel
    def process_single_deepseek_request(request_data):
        """Process a single DeepSeek request."""
        chunk_id, example_index, intermediate_response = request_data
        
        if not intermediate_response:  # Empty response from O3 error
            return chunk_id, example_index, f"{config.EVAL_ANSWER_PHRASE}"  # Return just answer phrase
        
        try:
            resp = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "user", "content": config.EVAL_QUESTION_PHRASE},
                    {"role": "assistant", "content": f"{intermediate_response}\n\n{config.EVAL_ANSWER_PHRASE}", "prefix": True}
                ],
                max_tokens=10,
                temperature=0
            )
            final_response = resp.choices[0].message.content
            # Combine intermediate response + answer phrase + final response (like other models)
            full_response = f"{intermediate_response}\n\n{config.EVAL_ANSWER_PHRASE}{final_response}"
            return chunk_id, example_index, full_response
        except Exception as e:
            logger.warning(f"DeepSeek API error for chunk {chunk_id} example {example_index}: {e}")
            return chunk_id, example_index, intermediate_response  # Fallback to intermediate only
    
    logger.info(f"Submitting {len(all_requests)} requests to DeepSeek in parallel...")
    
    # Use ThreadPoolExecutor for parallel API calls
    max_workers = min(50, len(all_requests))  # Limit concurrent requests to avoid rate limits
    final_responses_by_chunk = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all requests
        future_to_request = {
            executor.submit(process_single_deepseek_request, request_data): request_data
            for request_data in all_requests
        }
        
        # Collect results as they complete
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_request):
            try:
                chunk_id, example_index, final_response = future.result()
                
                # Organize responses by chunk
                if chunk_id not in final_responses_by_chunk:
                    final_responses_by_chunk[chunk_id] = {}
                final_responses_by_chunk[chunk_id][example_index] = final_response
                
                completed_count += 1
                if completed_count % 100 == 0:
                    logger.info(f"Completed {completed_count}/{len(all_requests)} DeepSeek requests")
                    
            except Exception as e:
                request_data = future_to_request[future]
                logger.error(f"DeepSeek request failed for {request_data}: {e}")
    
    # Step 3: Convert to ordered lists by chunk
    final_responses_ordered = {}
    for chunk_id, responses_dict in final_responses_by_chunk.items():
        # Sort by example_index to maintain order
        ordered_responses = [responses_dict[i] for i in sorted(responses_dict.keys())]
        final_responses_ordered[chunk_id] = ordered_responses
    
    logger.info(f"✅ Completed all {len(all_requests)} DeepSeek requests in parallel")
    
    # Save final responses
    final_responses_file = data.group_dir / 'final_responses.json'
    safe_file_operation(final_responses_file, 'write_json', final_responses_ordered)
    
    logger.info(f"Final responses saved: {final_responses_file}")
    return data.with_responses(final_responses_ordered)

def parse_openai_response_pure(response_body, model_name):
    """Parse OpenAI response format to extract reasoning and answer."""
    if model_name == 'o3':
        # O3 reasoning API format
        reasoning, answer = '', ''
        
        for output_item in response_body.get('output', []):
            if output_item.get('type') == 'reasoning':
                summary_texts = []
                for item in output_item.get('summary', []):
                    if isinstance(item, dict) and 'text' in item:
                        summary_texts.append(item['text'])
                    elif hasattr(item, 'text'):
                        summary_texts.append(item.text)
                reasoning = '\n\n'.join(summary_texts)
            elif output_item.get('type') == 'message':
                answer = next((item.get('text', '') for item in output_item.get('content', []) if item.get('type') == 'output_text'), '')
        
        return f"{reasoning}\n\n{answer}" if reasoning else answer
        
    elif model_name == 'gpt-4.1':
        # GPT-4.1 also uses reasoning API format (same as O3 but without detailed reasoning)
        answer = ''
        
        for output_item in response_body.get('output', []):
            if output_item.get('type') == 'message':
                answer = next((item.get('text', '') for item in output_item.get('content', []) if item.get('type') == 'output_text'), '')
        
        return answer
    else:
        raise ValueError(f"Unsupported model for response parsing: {model_name}")

def read_minimal_tracker(group_dir: Path):
    """Read minimal batch tracking info."""
    batch_file = group_dir / 'batches.jsonl'
    meta_file = group_dir / 'metadata.json'
    
    # Read batch info
    batch_infos = []
    if batch_file.exists():
        with open(batch_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                batch_infos.append(data)
    
    # Read metadata
    metadata = {}
    if meta_file.exists():
        metadata = safe_file_operation(meta_file, 'read_json')
    
    return batch_infos, metadata

def check_existing_group_status_pure(data: BatchData) -> BatchData:
    """Check status of existing group using minimal tracker and OpenAI API."""
    try:
        batch_infos, metadata = read_minimal_tracker(data.group_dir)
        
        if not batch_infos:
            logger.warning("No batch info found in minimal tracker")
            return data.with_metadata(resume_existing=False)
        
        # Check batch statuses via OpenAI API
        openai_client, _ = get_openai_clients()
        chunk_statuses = []
        
        logger.info(f"Checking status of {len(batch_infos)} batches via OpenAI API...")
        
        for batch_info in batch_infos:
            try:
                batch_status = openai_client.batches.retrieve(batch_info['batch_id'])
                chunk_statuses.append(batch_status.status)
                logger.debug(f"Batch {batch_info['chunk_id']}: {batch_status.status}")
            except Exception as e:
                logger.error(f"Error checking batch {batch_info['batch_id']}: {e}")
                chunk_statuses.append('error')
        
        all_completed = all(status == 'completed' for status in chunk_statuses)
        
        if all_completed:
            logger.info("✅ All chunks already completed, proceeding to DeepSeek processing")
            # Create tracker format compatible with processing
            tracker = {
                'chunks': {str(info['chunk_id']): {
                    'batch_id': info['batch_id'],
                    'status': 'completed',
                    'chunk_start_idx': info['chunk_start_idx'],
                    'chunk_end_idx': info['chunk_end_idx'],
                    'chunk_size': info['chunk_size']
                } for info in batch_infos}
            }
            return data.with_tracker(tracker).with_metadata(skip_to_deepseek=True)
        else:
            # Some chunks still in progress
            from collections import Counter
            status_counts = dict(Counter(chunk_statuses))
            logger.info(f"📊 Group status: {status_counts}")
            logger.info("⏳ Resuming monitoring of existing group...")
            
            # Create tracker for monitoring
            tracker = {
                'chunks': {str(info['chunk_id']): {
                    'batch_id': info['batch_id'],
                    'status': chunk_statuses[i],
                    'chunk_start_idx': info['chunk_start_idx'],
                    'chunk_end_idx': info['chunk_end_idx'],
                    'chunk_size': info['chunk_size']
                } for i, info in enumerate(batch_infos)}
            }
            return data.with_tracker(tracker).with_metadata(skip_to_deepseek=False)
            
    except (FileNotFoundError, KeyError, Exception) as e:
        logger.warning(f"Could not read minimal tracker: {e}")
        logger.info("🔄 Will resubmit group...")
        return data.with_metadata(resume_existing=False)

# =============================================================================
# PIPELINE CONFIGURATIONS
# =============================================================================

def create_submission_pipeline() -> List[PipelineStep]:
    """Create pipeline for new group submission."""
    return [
        PipelineStep("create_chunks", create_chunks_pure, description="Split examples into chunks"),
        PipelineStep("setup_group_dir", setup_group_directory_pure, description="Create group directory structure"),
        PipelineStep("handle_existing", handle_existing_group_pure, description="Handle existing group conflicts"),
        PipelineStep("create_directory", create_group_directory_pure, description="Create group directory"),
        PipelineStep("validate_sizes", validate_chunk_sizes_pure, description="Validate chunk sizes"),
        PipelineStep("submit_chunks", submit_chunks_pure, retry=True, description="Submit all chunks"),
        PipelineStep("wait_completion", wait_for_completion_pure, description="Wait for completion"),
        PipelineStep("process_deepseek", process_with_deepseek_pure, description="Process with DeepSeek")
    ]

def create_resume_pipeline() -> List[PipelineStep]:
    """Create pipeline for resuming existing group."""
    return [
        PipelineStep("create_chunks", create_chunks_pure, description="Split examples into chunks"),
        PipelineStep("setup_group_dir", setup_group_directory_pure, description="Create group directory structure"),
        PipelineStep("handle_existing", handle_existing_group_pure, description="Handle existing group conflicts"),
        PipelineStep("check_status", check_existing_group_status_pure, description="Check existing group status"),
        PipelineStep("wait_completion", wait_for_completion_pure, required=False, description="Wait if needed"),
        PipelineStep("process_deepseek", process_with_deepseek_pure, description="Process with DeepSeek")
    ]

def flatten_responses_pure(data: BatchData) -> List[str]:
    """Flatten chunk responses back to original order."""
    all_responses = [''] * len(data.examples)
    
    for chunk_info in data.chunks:
        chunk_responses = data.responses.get(chunk_info['chunk_id'], [])
        for i, response in enumerate(chunk_responses):
            all_responses[chunk_info['chunk_idx'] + i] = response
    
    return all_responses

# =============================================================================
# MAIN OPENAI O3 BATCH PROCESSING FUNCTION
# =============================================================================

def get_openai_responses(batch_examples, model_name, model_kwargs, mode_type, dataset_name='images'):
    """
    Generate responses using OpenAI batch API with functional pipeline architecture.
    
    Args:
        batch_examples: List of examples with 'messages' keys
        model_name: OpenAI model name ('o3', 'gpt-4.1')
        model_kwargs: Model generation parameters 
        mode_type: Evaluation mode 
        dataset_name: Dataset identifier for batch tracking
        
    Returns:
        List of response strings from OpenAI model (flattened from all chunks)
    """
    # Both O3 and GPT-4.1 support all modes since prompts are built in evaluate_images.py
    
    logger.info(f"Starting {model_name} batch evaluation for {len(batch_examples)} examples")
    
    # Create initial data structure
    config_obj = BatchConfig()
    initial_data = BatchData(
        examples=tuple(batch_examples),
        config=config_obj,
        metadata={
            'model_name': model_name,
            'model_kwargs': model_kwargs,
            'mode_type': mode_type,
            'dataset_name': dataset_name
        }
    )
    
    # Determine pipeline based on resume logic
    # First check if we need to resume
    temp_data = execute_pipeline(initial_data, [
        PipelineStep("create_chunks", create_chunks_pure),
        PipelineStep("setup_group_dir", setup_group_directory_pure),
        PipelineStep("handle_existing", handle_existing_group_pure)
    ])
    
    # Choose pipeline based on resume decision
    if temp_data.metadata.get('resume_existing', False):
        logger.info("Resuming existing group")
        pipeline = create_resume_pipeline()[3:]  # Skip already executed steps
        final_data = execute_pipeline(temp_data, pipeline)
        
        # Skip to DeepSeek if already completed
        if temp_data.metadata.get('skip_to_deepseek', False):
            final_data = process_with_deepseek_pure(temp_data)
    else:
        logger.info("Creating new group")
        pipeline = create_submission_pipeline()[3:]  # Skip already executed steps
        final_data = execute_pipeline(temp_data, pipeline)
    
    # Flatten and return responses
    return flatten_responses_pure(final_data)