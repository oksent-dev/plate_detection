import cv2
import numpy as np
from easyocr import Reader


def predict_plate_number(image_path):
    image = cv2.imread(image_path)
    orig = image.copy()
    reader = Reader(["en"])
    initial_results = reader.readtext(image, width_ths=0.8, canvas_size=5000)
    valid_candidates = find_best_plate_candidates(initial_results)

    if valid_candidates:
        best_candidate = None
        best_score = float("inf")
        best_processed_roi = None

        for candidate in valid_candidates:
            (tl, tr, br, bl) = candidate[0]
            x_min = int(min(tl[0], bl[0]))
            y_min = int(min(tl[1], tr[1]))
            x_max = int(max(tr[0], br[0]))
            y_max = int(max(bl[1], br[1]))

            plate_roi = orig[y_min:y_max, x_min:x_max]
            if plate_roi.shape[1] < 60 or plate_roi.shape[0] < 20:
                continue
            processed_roi = preprocess_roi(plate_roi)

            final_results = reader.recognize(
                processed_roi,
                allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            )
            if final_results:
                best_final = max(final_results, key=lambda x: x[2])
                plate_text = correct_plate_text(best_final[1])
                if not is_valid_plate(plate_text):
                    continue
                score = abs(len(plate_text) - 7)
                if score < best_score:
                    best_score = score
                    best_candidate = plate_text
                    best_processed_roi = processed_roi

        if best_candidate:
            return best_candidate, best_processed_roi

    return "", None


def correct_plate_text(text):
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

    if text and text[0] in ["I", "1"]:
        text = text[1:]

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


def preprocess_roi(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    kernel = None
    if gray.shape[1] > 400:
        kernel = np.ones((7, 7), np.uint8)
    elif gray.shape[1] <= 400 and gray.shape[1] > 225:
        kernel = np.ones((3, 3), np.uint8)
    if kernel is not None:
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    gray = cv2.bitwise_not(gray)
    return gray


def can_be_valid_plate(text):
    if 4 <= len(text) <= 11:
        return True
    return False


def is_valid_plate(text):
    forbidden = ["GREELJR", "GLIWICE", "GLIW1CE", "GUIWICE", "GUIW1CE"]
    if can_be_valid_plate(text) and text not in forbidden:
        return True
    return False


def find_best_plate_candidates(results):
    valid_candidates = []
    for detection in sorted(results, key=lambda x: -x[2]):
        text = detection[1]
        if can_be_valid_plate(text):
            valid_candidates.append(detection)
    return valid_candidates
