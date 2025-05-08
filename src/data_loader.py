import xml.etree.ElementTree as ET
from collections import defaultdict
import os
import cv2
from tqdm import tqdm
from src.config import config


def parse_annotations(xml_path: str) -> dict:
    """
    Parses the XML annotations file to extract image filenames and actual plate numbers.

    Parameters:
    - xml_path: Path to the XML annotations file.

    Returns:
    - annotations: Dictionary with image filenames as keys and actual plate numbers as values.
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"Annotations file not found at {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = defaultdict(str)
    for image in root.findall("image"):
        filename = image.get("name")
        for box in image.findall("box"):
            if box.get("label") == "plate":
                plate_number = box.find("attribute[@name='plate number']").text
                annotations[filename] = plate_number
    return annotations


def load_images(annotations: dict) -> dict:
    """
    Loads images from the dataset based on the provided annotations.

    Parameters:
    - annotations: Dictionary of annotations containing image filenames and actual plate numbers.

    Returns:
    - images: Dictionary of loaded images with filenames as keys.
    """
    images = {}
    for filename in tqdm(annotations.keys(), desc="Loading images"):
        img_path = os.path.join(config.RAW_IMAGES_PATH, filename)
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_REDUCED_COLOR_2)
            if img is not None:
                images[filename] = img
    return images
