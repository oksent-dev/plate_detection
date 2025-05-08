import numpy as np
import os
from ultralytics import YOLO
from src.config import Config
from paddleocr import PaddleOCR
from src.const import police_plates

config = Config()
ocr = PaddleOCR(lang="en", show_log=False, use_gpu=True)
model_path = (
    config.PROJECT_ROOT / "src" / "runs" / "detect" / "train" / "weights" / "best.pt"
)
if not os.path.exists(model_path):
    print("Model not found. Please train the model first.")
    exit(1)

model = YOLO(model_path)


def predict_plate_number(image: np.ndarray) -> tuple[str, np.ndarray]:
    """
    Predicts the plate number from the given image using YOLOv8 and PaddleOCR.

    Parameters:
    - image: Input image in which to detect the plate number.

    Returns:
    - (plate_text, processed_roi): Tuple containing the predicted plate text and the ROI of plate.
    """

    results = model.predict(image, verbose=False, conf=0.02)
    if not results or not results[0].boxes:
        return "", None

    best_conf = 0
    for result in results:
        for box in result.boxes:
            if box.conf > best_conf:
                best_conf = box.conf
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                h = y2 - y1
                w = x2 - x1
                if h < 33:
                    margin = 5
                    y1 = max(0, y1 - margin)
                    y2 = min(image.shape[0], y2 + margin)
                if w < 151:
                    margin = 25
                    x1 = max(0, x1)
                    x2 = min(image.shape[1], x2 + margin)
                elif w < 175:
                    margin = 20
                    x2 = min(image.shape[1], x2 + margin)
                plate_roi = image[y1:y2, x1:x2]

    processed_roi = plate_roi

    results = ocr.ocr(
        processed_roi,
        cls=False,
        det=False,
    )
    if results and len(results[0]) > 0:
        best_result = max(results[0], key=lambda x: x[1])
        plate_text = correct_plate_text(best_result[0])
        return plate_text, processed_roi

    return "", processed_roi


def correct_plate_text(text: str) -> str:
    """
    Corrects the common mistakes in the plate text.

    Parameters:
    - text: plate text to be corrected.

    Returns:
    - corrected text.
    """
    number_to_letter = {
        "5": "S",
        "2": "Z",
        "0": "O",
        "8": "B",
    }
    letter_to_number = {
        "B": "8",
        "D": "0",
        "I": "1",
        "O": "0",
        "Z": "7",
    }
    text = "".join(filter(str.isalnum, text))
    text = text.upper()

    # Plate cannot start with those letters
    if text and text[0] in ["I", "1", "A"]:
        text = text[1:]

    # If plate starts with 'H' and is not a police plate, it must be false positive
    if text and text[0] == "H":
        if text[0:3] not in police_plates:
            text = text[1:]

    # If plate starts with SC and is 8 characters long, it must have been confused with SCI
    if text[0:2] == "SC" and len(text[2:]) == 6 and text[2] == "1":
        text = text[:2] + "I" + text[3:]

    # Plate with 2 letter code can be max 7 characters long
    if text[0:2].isalpha() and text[2].isdigit() and len(text) == 8:
        text = text[:-1]

    # Combination of 2 letters + 3 digits + 1 letter + 1 digit cannot occur, probably 'I' confused with '1'
    if (
        text[0:2].isalpha()
        and text[2:5].isdigit()
        and text[5].isalpha()
        and text[6].isdigit()
        and len(text) == 7
    ):
        text = text[:2] + "I" + text[3:]

    corrected = []

    for i, char in enumerate(text):
        if i == 0:
            corrected.append(number_to_letter.get(char, char))
        elif i >= 3:
            corrected.append(letter_to_number.get(char, char))
        else:
            corrected.append(char)

    final_text = "".join(corrected)

    return final_text[:8]
