# experiments/load_d3.py
import os
import pandas as pd
from PIL import Image, UnidentifiedImageError
import tqdm
import requests
import io
import logging
from pathlib import Path
from typing import Union, List, Dict, Set, Any
import argparse
import sys
import ast # <--- Added import for ast.literal_eval

# --- New Import ---
from datasets import load_dataset

# Assuming config.py is in the parent directory of 'experiments'
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
import config

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=config.LOAD_D3_LOG_FILE,
    filemode='a'
)

# Constants
URL_COL = 'url' # Field name for the real image URL in HF dataset item
# Field names for AI-generated images in HF dataset item
GEN_IMAGE_KEYS = [f'image_gen{i}' for i in range(4)] 

def _save_pil_image(pil_image: Image.Image, save_path: Path, item_identifier_for_log: str, image_type_for_log: str) -> bool:
    """Helper to save a PIL image."""
    try:
        rgb_img = pil_image.convert("RGB")
        rgb_img.save(save_path)
        logging.debug(f"{item_identifier_for_log}: Saved {image_type_for_log} image to {save_path.name}")
        return True
    except Exception as e_save:
        logging.error(f"{item_identifier_for_log}: Could not save {image_type_for_log} image {save_path.name}. Error: {e_save}")
        return False

def _process_single_hf_item(
    hf_item_data: Dict[str, Any],
    csv_row_index: int, 
    save_dir: Path,
    http_session: requests.Session,
    timeout: int,
    overwrite: bool
) -> bool:
    """Processes a single item from Hugging Face data, using csv_row_index for naming."""
    actual_hf_id = str(hf_item_data.get('id', 'unknown_hf_id'))
    item_identifier_for_log = f"CSVIndex_{csv_row_index}(HF_ID_{actual_hf_id})"
    all_images_processed_successfully = True # Assume success until a failure

    # --- 1. Process Real Image (from URL) ---
    real_img_url = hf_item_data.get(URL_COL)
    real_img_path = save_dir / f"{csv_row_index}_real.png"

    if not real_img_url:
        logging.error(f"{item_identifier_for_log}: Real image URL ('{URL_COL}') not found or empty in HF item.")
        # Consider this a failure for the real image part, but AI images might still be processed.
        # Depending on requirements, you might want to return False immediately.
        # For now, we'll mark it and let AI images try.
        all_images_processed_successfully = False 
    elif not overwrite and real_img_path.exists():
        logging.debug(f"{item_identifier_for_log}: Real image {real_img_path.name} already exists. Skipping.")
    else:
        saved_real = False
        try:
            response = http_session.get(real_img_url, stream=True, timeout=timeout)
            response.raise_for_status()
            with Image.open(response.raw) as downloaded_real_img:
                saved_real = _save_pil_image(downloaded_real_img, real_img_path, item_identifier_for_log, "real (from URL)")
        except requests.exceptions.RequestException as e_req:
            logging.error(f"{item_identifier_for_log}: Failed to download real image from URL {real_img_url}. Error: {e_req}")
        except UnidentifiedImageError as e_img:
            logging.error(f"{item_identifier_for_log}: Failed to identify real image from URL {real_img_url}. Error: {e_img}")
        except Exception as e_real: # Catch any other errors during real image processing
            logging.error(f"{item_identifier_for_log}: Error processing real image from URL {real_img_url}. Error: {e_real}")
        
        if not saved_real:
            all_images_processed_successfully = False

    # --- 2. Process AI-Generated Images (from 'image_genX' fields) ---
    for ai_image_key in GEN_IMAGE_KEYS: # e.g., "image_gen0", "image_gen1", ...
        raw_ai_image_data = hf_item_data.get(ai_image_key)

        if raw_ai_image_data is None:
            logging.debug(f"{item_identifier_for_log}: No data found for AI image key '{ai_image_key}'. Skipping this AI image.")
            continue # Skip to the next AI image key

        # Filename stem is the key itself (image_gen0, etc.)
        gen_img_path = save_dir / f"{csv_row_index}_{ai_image_key}.png"

        if not overwrite and gen_img_path.exists():
            logging.debug(f"{item_identifier_for_log}: Generated image {gen_img_path.name} already exists. Skipping.")
            continue

        pil_image_to_save = None
        try:
            if isinstance(raw_ai_image_data, Image.Image):
                pil_image_to_save = raw_ai_image_data
            elif isinstance(raw_ai_image_data, dict) and 'bytes' in raw_ai_image_data and isinstance(raw_ai_image_data['bytes'], bytes):
                with io.BytesIO(raw_ai_image_data['bytes']) as img_byte_stream:
                    pil_image_to_save = Image.open(img_byte_stream)
            elif isinstance(raw_ai_image_data, str):
                image_dict = ast.literal_eval(raw_ai_image_data) # Can raise ValueError, SyntaxError
                if isinstance(image_dict, dict) and 'bytes' in image_dict and isinstance(image_dict['bytes'], bytes):
                    with io.BytesIO(image_dict['bytes']) as img_byte_stream:
                        pil_image_to_save = Image.open(img_byte_stream) # Can raise UnidentifiedImageError
                else:
                    raise ValueError("Parsed string was not a dict with 'bytes' key holding bytes.")
            else:
                logging.error(f"{item_identifier_for_log}: AI image data for {ai_image_key} is in an unrecognized format: {type(raw_ai_image_data)}. Skipping.")
                all_images_processed_successfully = False
                continue

            if pil_image_to_save:
                if not _save_pil_image(pil_image_to_save, gen_img_path, item_identifier_for_log, f"AI-generated ({ai_image_key})"):
                    all_images_processed_successfully = False
            else: # Should only be reached if a format was recognized but failed conversion to PIL (e.g. bad bytes dict)
                if not isinstance(raw_ai_image_data, Image.Image): # ensure it wasn't already a PIL image that failed saving
                     logging.error(f"{item_identifier_for_log}: Failed to obtain a processable PIL image for {ai_image_key} from raw data.")
                     all_images_processed_successfully = False

        except (ValueError, SyntaxError) as e_eval: # For ast.literal_eval
            logging.error(f"{item_identifier_for_log}: Could not parse AI image data string for {ai_image_key}. Data: '{str(raw_ai_image_data)[:100]}...'. Error: {e_eval}")
            all_images_processed_successfully = False
        except UnidentifiedImageError as e_unid: # For Image.open
            logging.error(f"{item_identifier_for_log}: Could not identify image from bytes for {ai_image_key}. Error: {e_unid}")
            all_images_processed_successfully = False
        except Exception as e_proc_ai: # Catch-all for this AI image
            logging.error(f"{item_identifier_for_log}: Unexpected error processing AI image {ai_image_key}. Error: {e_proc_ai}")
            all_images_processed_successfully = False
            
    return all_images_processed_successfully


def load_d3_images_from_hf_in_order(
    ids_csv_filepath: Union[str, Path],
    id_column_name: str,
    save_directory: Union[str, Path],
    timeout: int = 10,
    overwrite: bool = False
) -> None:
    ids_csv_file = Path(ids_csv_filepath)
    save_dir = Path(save_directory)

    if not ids_csv_file.is_file():
        logging.error(f"ID CSV file not found: {ids_csv_file}")
        return

    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Using save directory: {save_dir}")
    except OSError as e:
        logging.error(f"Could not create save directory {save_dir}. Error: {e}")
        return

    try:
        ids_df = pd.read_csv(ids_csv_file)
        if id_column_name not in ids_df.columns:
            logging.error(f"ID column '{id_column_name}' not found in {ids_csv_file}. Cols: {ids_df.columns.tolist()}")
            return
        ordered_processing_ids_from_csv: List[str] = ids_df[id_column_name].astype(str).tolist()
        target_ids_to_cache: Set[str] = set(ordered_processing_ids_from_csv)
        logging.info(f"Loaded {len(ordered_processing_ids_from_csv)} IDs from {ids_csv_file} (Unique: {len(target_ids_to_cache)}). Will process in this order.")
    except Exception as e:
        logging.error(f"Error reading ID CSV file {ids_csv_file}. Error: {e}")
        return

    logging.info(f"Starting caching phase: Looking for {len(target_ids_to_cache)} unique IDs in the 'elsaEU/ELSA_D3' stream...")
    hf_items_cache: Dict[str, Dict[str, Any]] = {}
    
    try:
        elsa_data_stream = load_dataset("elsaEU/ELSA_D3", split="validation", streaming=True)
    except Exception as e:
        logging.error(f"Failed to load Hugging Face dataset 'elsaEU/ELSA_D3'. Error: {e}")
        return

    stream_iterator = tqdm.tqdm(elsa_data_stream, desc="Caching items from ELSA_D3 stream")
    for hf_item in stream_iterator:
        hf_item_id_str = str(hf_item.get('id', ''))
        if hf_item_id_str in target_ids_to_cache:
            if hf_item_id_str not in hf_items_cache:
                hf_items_cache[hf_item_id_str] = hf_item
                stream_iterator.set_postfix_str(f"Cached {len(hf_items_cache)}/{len(target_ids_to_cache)} items")
            if len(hf_items_cache) == len(target_ids_to_cache):
                logging.info("All required unique IDs found and cached.")
                break 
    logging.info(f"Caching phase finished. Cached {len(hf_items_cache)} unique items.")

    logging.info(f"Starting processing phase for {len(ordered_processing_ids_from_csv)} IDs in specified order...")
    processed_items_fully_successful = 0
    items_with_some_errors = 0
    not_found_in_cache_ids: List[str] = []

    with requests.Session() as http_session:
        for csv_idx, current_hf_id_to_process in enumerate(tqdm.tqdm(ordered_processing_ids_from_csv, desc="Processing cached items")):
            if current_hf_id_to_process not in hf_items_cache:
                logging.warning(f"ID {current_hf_id_to_process} (CSV row {csv_idx}) was not found in the stream cache. Skipping.")
                if current_hf_id_to_process not in not_found_in_cache_ids:
                    not_found_in_cache_ids.append(current_hf_id_to_process)
                continue
            
            hf_item_data = hf_items_cache[current_hf_id_to_process]
            if _process_single_hf_item(hf_item_data, csv_idx, save_dir, http_session, timeout, overwrite):
                processed_items_fully_successful += 1
            else:
                items_with_some_errors += 1
                logging.info(f"Item for HF ID {current_hf_id_to_process} (CSV row {csv_idx}) processed with one or more errors.")

    logging.info(f"Processing phase finished.")
    logging.info(f"Items fully processed successfully: {processed_items_fully_successful}")
    logging.info(f"Items processed with one or more errors: {items_with_some_errors}")
    if not_found_in_cache_ids:
        logging.warning(f"The following {len(not_found_in_cache_ids)} unique IDs from CSV were not found in the dataset stream: {not_found_in_cache_ids}")
    
    unique_ids_in_csv_count = len(target_ids_to_cache)
    ids_actually_in_cache_count = len(hf_items_cache)
    if ids_actually_in_cache_count < unique_ids_in_csv_count:
         logging.warning(f"Note: {unique_ids_in_csv_count - ids_actually_in_cache_count} unique IDs from the CSV were not found in the stream at all.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download/save D3 images for specified IDs from 'elsaEU/ELSA_D3' dataset, in CSV order, using CSV row index for naming."
    )
    parser.add_argument('--ids_csv_filepath', type=Path, default=config.D3_CSV_FILE,
                        help=f"Path to CSV file with IDs. Default: {config.D3_CSV_FILE}")
    parser.add_argument('--id_column_name', type=str, default='id',
                        help="Name of ID column in CSV. Default: 'id'")
    parser.add_argument('--save_directory', type=Path, default=config.D3_DIR,
                        help=f"Directory to save images. Default: {config.D3_DIR}")
    parser.add_argument("-t", "--timeout", type=int, default=15,
                        help="Timeout for downloads (seconds). Default: 15")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Force overwrite of existing files.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging.")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    logging.info(f"Starting D3 image loading (ordered by CSV, names by CSV index).")
    logging.info(f"ID CSV file: {args.ids_csv_filepath}")
    logging.info(f"ID column: {args.id_column_name}")
    logging.info(f"Save directory: {args.save_directory}")
    if args.force: logging.info("Overwrite mode enabled.")

    load_d3_images_from_hf_in_order(
        ids_csv_filepath=args.ids_csv_filepath,
        id_column_name=args.id_column_name,
        save_directory=args.save_directory,
        timeout=args.timeout,
        overwrite=args.force
    )
    logging.info("D3 image loading process finished.")