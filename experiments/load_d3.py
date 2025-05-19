import os
import pandas as pd
from PIL import Image, UnidentifiedImageError # Import specific error
import tqdm
import requests
import io
import ast
import logging # Use logging instead of print for errors/info
from pathlib import Path # Use pathlib for path manipulation
from typing import Union # For type hinting paths

# --- Configuration ---
# Configure logging for better feedback
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='processing_log.log', # Add this line
    filemode='a' # Optional: 'a' for append (default), 'w' for overwrite
)

# Define columns containing generated image data more dynamically
GEN_IMAGE_COLS = [f'image_gen{i}' for i in range(4)]
URL_COL = 'url' # Define column names as constants

# --- Function Definition ---
def load_d3_data(
    csv_filepath: Union[str, Path],
    save_directory: Union[str, Path],
    timeout: int = 10,
    overwrite: bool = False
) -> None:
    """
    Downloads real images and extracts/saves generated images from a CSV file.
    If any error occurs during the processing of ANY image (real or generated)
    for a given row, the entire row processing is skipped.

    Args:
        csv_filepath: Path to the input CSV file.
                       Expected columns: 'url', 'image_gen0', ..., 'image_gen3'.
        save_directory: Directory where images will be saved. It will be created if it doesn't exist.
        timeout: Timeout in seconds for the image download request.
        overwrite: If True, overwrite existing image files. Otherwise, skip them.
    """
    csv_file = Path(csv_filepath)
    save_dir = Path(save_directory)

    # --- Pre-checks and Setup ---
    if not csv_file.is_file():
        logging.error(f"CSV file not found: {csv_file}")
        return

    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Using save directory: {save_dir}")
    except OSError as e:
        logging.error(f"Could not create save directory {save_dir}. Error: {e}")
        return

    try:
        data = pd.read_csv(csv_file)
        logging.info(f"Successfully loaded {len(data)} rows from {csv_file}")
    except pd.errors.EmptyDataError:
        logging.error(f"CSV file is empty: {csv_file}")
        return
    except Exception as e:
        logging.error(f"Error reading CSV file {csv_file}. Error: {e}")
        return

    # --- Main Processing Loop ---
    processed_rows = 0
    skipped_rows = 0
    with requests.Session() as session:
        for i, row in tqdm.tqdm(data.iterrows(), total=len(data), desc="Processing rows"):
            row_identifier = f"Row {i}"
            try: # <<< Single try block encompassing all processing for the row
                # --- 1. Process Real Image from URL ---
                real_img_path = save_dir / f"{i}_real.png"
                # Check existence only if not overwriting
                if not overwrite and real_img_path.exists():
                    logging.debug(f"{row_identifier}: Real image {real_img_path.name} already exists. Skipping download.")
                else:
                    # Check if URL is present
                    if URL_COL not in row or pd.isna(row[URL_COL]):
                        # Treat missing URL as an error preventing row completion
                        raise ValueError("URL is missing or NaN, cannot process real image.")

                    url = row[URL_COL]
                    # Perform download and save (these can raise exceptions)
                    response = session.get(url, stream=True, timeout=timeout)
                    response.raise_for_status()  # Raises HTTPError for bad responses

                    with Image.open(response.raw) as real_img: # Raises UnidentifiedImageError
                        rgb_img = real_img.convert("RGB")
                        rgb_img.save(real_img_path) # Raises OSError, IOError
                    logging.debug(f"{row_identifier}: Saved real image to {real_img_path.name}")

                # --- 2. Process AI-Generated Images ---
                for col in GEN_IMAGE_COLS:
                    gen_img_path = save_dir / f"{i}_{col}.png"
                    # Check existence only if not overwriting
                    if not overwrite and gen_img_path.exists():
                        logging.debug(f"{row_identifier}: Generated image {gen_img_path.name} already exists. Skipping this generated image.")
                        continue # Skips only this generated image, moves to next col

                    # Check if column/data exists
                    if col not in row or pd.isna(row[col]):
                        # Treat missing data as an error preventing row completion
                        raise ValueError(f"No data in column {col}, cannot process generated image.")

                    # Extract, decode, and save (these can raise exceptions)
                    cell_content = str(row[col])
                    image_dict = ast.literal_eval(cell_content) # Raises ValueError, SyntaxError

                    if not isinstance(image_dict, dict) or 'bytes' not in image_dict:
                        raise TypeError(f"Invalid structure in column {col}. Expected dict with 'bytes' key.")

                    image_data = image_dict['bytes']
                    if not isinstance(image_data, bytes):
                        raise TypeError(f"Data in 'bytes' key of column {col} is not bytes type.")

                    with io.BytesIO(image_data) as img_byte_stream:
                        with Image.open(img_byte_stream) as img: # Raises UnidentifiedImageError
                            rgb_img = img.convert("RGB")
                            rgb_img.save(gen_img_path) # Raises OSError, IOError
                    logging.debug(f"{row_identifier}: Saved generated image to {gen_img_path.name}")

                # If we reach here, the row was processed successfully without errors
                processed_rows += 1

            # --- Catch ANY exception during the row's processing ---
            except (requests.exceptions.RequestException,  # Includes Timeout, ConnectionError, HTTPError
                    UnidentifiedImageError,
                    OSError, IOError,             # File saving/opening errors
                    ValueError, SyntaxError,      # ast.literal_eval errors or missing data errors we raised
                    TypeError,                    # Data structure/type errors we raised
                    KeyError,                     # Should not happen with checks, but just in case
                    Exception) as e:              # Catch any other unexpected exceptions
                logging.warning(f"{row_identifier}: Skipping entire row due to error: {e.__class__.__name__}: {e}")
                skipped_rows += 1
                continue # <<< Skips the rest of this iteration, moves to the next row

    logging.info(f"Processing summary: Successfully processed rows: {processed_rows}, Skipped rows due to errors: {skipped_rows}")

# --- Example Usage ---
if __name__ == "__main__":
    # Use argparse for command-line arguments for flexibility
    import argparse

    parser = argparse.ArgumentParser(description="Download real images and extract generated images specified in a CSV.")
    parser.add_argument("-t", "--timeout", type=int, default=10, help="Timeout for image downloads in seconds (default: 10).")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite of existing image files.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")

    args = parser.parse_args()

    # Adjust logging level if verbose flag is set
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    load_d3_data(
        csv_filepath="/data3/singhdan/ELSA_D3/D3_2k_sample.csv",
        save_directory="/data3/zkachwal/ELSA_D3",
        timeout=args.timeout,
        overwrite=args.force
    )

    logging.info("Processing finished.")