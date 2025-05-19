import json
import os

def load_scores_from_jsonl(file_path):
    """
    Loads image scores from a JSON file, expecting a JSON array of records.

    Each record in the array is expected to be a JSON object containing
    at least "image" (path) and "cur_score" (integer).

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: A dictionary mapping image paths to their cur_score.
              Returns an empty dictionary if the file cannot be read, is not a JSON array,
              or contains no valid records.
    """
    scores = {}
    try:
        with open(file_path, 'r') as f:
            # Attempt to load the entire file as one JSON structure (expecting a list of records)
            all_records = json.load(f)

        if not isinstance(all_records, list):
            print(f"Error: Content of {file_path} is not a JSON list as expected. Found type: {type(all_records)}.")
            print("The script expects the file to contain a JSON array, like: [{\"key\": \"val\", ...}, {\"key\": \"val\", ...}]")
            return scores # Return empty scores

        print(f"Successfully parsed {file_path} as a JSON array with {len(all_records)} records.")

        for record_number, record in enumerate(all_records, 1):
            try:
                if not isinstance(record, dict):
                    print(f"Warning: Record {record_number} in {file_path} is not a JSON object (dict), but got {type(record)}. Skipping this record.")
                    continue

                image_path = record.get("image")
                cur_score = record.get("cur_score")

                if image_path is None or cur_score is None:
                    print(f"Warning: Skipping record {record_number} in {file_path} due to missing 'image' or 'cur_score' key. Record: {str(record)[:100]}...")
                    continue
                
                if not isinstance(image_path, str):
                    print(f"Warning: Skipping record {record_number} in {file_path}. 'image' path is not a string: {image_path}. Record: {str(record)[:100]}...")
                    continue

                # Ensure cur_score is an integer
                if isinstance(cur_score, str) and cur_score.isdigit():
                    cur_score = int(cur_score)
                elif not isinstance(cur_score, int):
                     print(f"Warning: Skipping record {record_number} in {file_path} for image '{image_path}' due to non-integer 'cur_score': {cur_score} (type: {type(cur_score)}).")
                     continue

                scores[image_path] = cur_score
            except Exception as e:
                # This catches errors during processing of an individual record after successful file parse
                print(f"Warning: An unexpected error occurred processing record {record_number} from {file_path}: {e}. Record snippet: '{str(record)[:100]}...'")

    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode {file_path} as a single JSON document. The file is not a valid JSON array or object.")
        print(f"JSONDecodeError details: {e}")
        print("Please ensure the file contains a valid JSON array, e.g., [{\"entry1\": ...}, {\"entry2\": ...}].")
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
    except Exception as e:
        # This catches other errors like permission issues during file open/read
        print(f"Error: Could not read or process file {file_path}: {e}")
    
    return scores

def find_images_with_specific_scores(scores_m1, scores_m2, scores_m3):
    """
    Finds images where method1 (zeroshot) score is 0, 
    method2 (zeroshot-cot) score is 0, and 
    method3 (zeroshot-2-artifacts) score is 1.

    Args:
        scores_m1 (dict): Dictionary of image_path: score for method 1 (zeroshot).
        scores_m2 (dict): Dictionary of image_path: score for method 2 (zeroshot-cot).
        scores_m3 (dict): Dictionary of image_path: score for method 3 (zeroshot-2-artifacts).

    Returns:
        list: A list of image paths matching the criteria.
    """
    matching_images = []

    # Get a set of all unique image paths from all methods
    # Ensure keys are strings, as image paths should be.
    all_image_paths = set()
    if scores_m1: # Check if dictionary is not empty
        all_image_paths.update(k for k in scores_m1.keys() if isinstance(k, str))
    if scores_m2:
        all_image_paths.update(k for k in scores_m2.keys() if isinstance(k, str))
    if scores_m3:
        all_image_paths.update(k for k in scores_m3.keys() if isinstance(k, str))


    if not all_image_paths:
        print("No valid image paths found in any of the loaded scores to compare.")
        return matching_images

    processed_count = 0
    for image_path in all_image_paths:
        processed_count +=1
        score_m1 = scores_m1.get(image_path)
        score_m2 = scores_m2.get(image_path)
        score_m3 = scores_m3.get(image_path)

        # Check if all scores are available (not None) and match the criteria
        if score_m1 == 0 and \
           score_m2 == 0 and \
           score_m3 == 1:
            matching_images.append(image_path)
    
    print(f"Processed {processed_count} unique image paths for score comparison.")
    return matching_images

def main():
    """
    Main function to execute the script.
    """
    # --- Configuration: Base path for the data files ---
    base_path = "/data3/zkachwal/visual-reasoning/data/ai-generation/responses/"
    
    # --- Define file names for each method ---
    # Method 1: zeroshot
    file_name_method1 = "AI_util-df402k-llama3-11b-zeroshot-n1-wait0-rationales.jsonl"
    # Method 2: zeroshot-cot
    file_name_method2 = "AI_util-df402k-llama3-11b-zeroshot-cot-n1-wait0-rationales.jsonl"
    # Method 3: zeroshot-2-artifacts
    file_name_method3 = "AI_util-df402k-llama3-11b-zeroshot-2-artifacts-n1-wait0-rationales.jsonl"

    # Construct full file paths
    file_path_method1 = os.path.join(base_path, file_name_method1)
    file_path_method2 = os.path.join(base_path, file_name_method2)
    file_path_method3 = os.path.join(base_path, file_name_method3)
    # ---------------------------------------------

    print(f"Attempting to load scores for Method 1 (zeroshot) from: {file_path_method1}")
    scores_method1 = load_scores_from_jsonl(file_path_method1)
    print(f"Loaded {len(scores_method1)} scores for Method 1.\n")

    print(f"Attempting to load scores for Method 2 (zeroshot-cot) from: {file_path_method2}")
    scores_method2 = load_scores_from_jsonl(file_path_method2)
    print(f"Loaded {len(scores_method2)} scores for Method 2.\n")

    print(f"Attempting to load scores for Method 3 (zeroshot-2-artifacts) from: {file_path_method3}")
    scores_method3 = load_scores_from_jsonl(file_path_method3)
    print(f"Loaded {len(scores_method3)} scores for Method 3.\n")

    # Check if any scores were loaded at all
    if not scores_method1 and not scores_method2 and not scores_method3:
        # Check if files actually exist to differentiate between "file not found" and "file empty/invalid"
        files_exist = [os.path.exists(p) for p in [file_path_method1, file_path_method2, file_path_method3]]
        if not any(files_exist):
             print("Critical Warning: None of the specified data files were found. Please check the file paths and names.")
        else:
             print("Warning: No scores were loaded from any of the files. This could be due to files being empty, not valid JSON arrays, or containing no records with 'image' and 'cur_score'.")
        
        print("Halting execution as no data is available to process.")
        return

    print("Finding matching images...")
    result_images = find_images_with_specific_scores(scores_method1, scores_method2, scores_method3)

    if result_images:
        print(f"\nFound {len(result_images)} images matching the criteria (zeroshot=0, zeroshot-cot=0, zeroshot-2-artifacts=1):")
        for image_path in result_images:
            print(image_path)
    else:
        print("\nNo images found matching the specified criteria.")

if __name__ == "__main__":
    main()
