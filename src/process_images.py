import cv2
import os
import time
import numpy as np
from tqdm import tqdm
from src.plate_detection import predict_plate_number
from src.config import config


def process_single_image(filename: str, actual_plate: str, img: np.ndarray) -> tuple:
    """
    Processes a single image to predict the license plate number.

    Parameters:
    - filename: Name of the image file.
    - actual_plate: Actual license plate number for comparison.
    - img: Image array.

    Returns:
    - filename: Name of the image file.
    - (predicted_plate, actual_plate): Tuple containing predicted and actual plate numbers.
    """
    predicted_plate, processed_image = predict_plate_number(img)

    if config.SAVE_IMAGES and processed_image is not None:
        processed_image_path = os.path.join(
            config.PROCESSED_IMAGES_PATH, f"processed_{filename}"
        )
        cv2.imwrite(processed_image_path, processed_image)

    return filename, (predicted_plate, actual_plate)


def process_images(images: dict, annotations: dict) -> tuple:
    """
    Processes images and predicts license plate numbers.

    Parameters:
    - images: Dictionary of images loaded from the dataset.
    - annotations: Dictionary of annotations containing actual plate numbers.

    Returns:
    - results: Dictionary containing predicted and actual plate numbers.
    - total_time: Total time taken to process all images.
    """
    results = {}
    start_time = time.time()
    for filename, actual_plate in tqdm(annotations.items(), desc="Processing images"):
        if filename in images:
            _, result = process_single_image(filename, actual_plate, images[filename])
            results[filename] = result
    total_time = time.time() - start_time
    return results, total_time
