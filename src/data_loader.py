import xml.etree.ElementTree as ET
from collections import defaultdict


def parse_annotations(xml_path):
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
