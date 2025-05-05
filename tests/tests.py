import os
from tqdm import tqdm
import pandas as pd
import cv2
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.plate_detection import predict_plate_number
from src.evaluation import evaluate_results
from pathlib import Path


def setup_test_environment():
    raw_images_path = Path("data/raw")
    processed_images_path = Path("data/processed")
    outputs_path = Path("outputs")

    raw_images_path.mkdir(parents=True, exist_ok=True)
    processed_images_path.mkdir(parents=True, exist_ok=True)
    outputs_path.mkdir(parents=True, exist_ok=True)

    return raw_images_path, processed_images_path, outputs_path


def load_test_annotations():
    return {
        # "1.jpg": "SCZ26114",
        # "10.jpg": "SK404XK",
        # "100.jpg": "SK321LE",
        # "106.jpg": "SH9264A",
        "5.jpg": "SRB9NP9",
        "15.jpg": "KOS8900E",
    }


def main():
    raw_images_path, processed_images_path, outputs_path = setup_test_environment()

    annotations = load_test_annotations()
    results = {}
    for filename, actual_plate in tqdm(annotations.items()):
        img_path = os.path.join(raw_images_path, filename)
        if not os.path.exists(img_path):
            print(f"Image {filename} not found in {raw_images_path}. Skipping...")
            continue

        predicted_plate, processed_image = predict_plate_number(img_path)
        if processed_image is not None:
            processed_image_path = os.path.join(
                processed_images_path, "processed_" + filename
            )
            cv2.imwrite(processed_image_path, processed_image)

        results[filename] = (predicted_plate, actual_plate)

    if not results:
        print("No results to evaluate.")
        return

    accuracy, differences = evaluate_results(results)

    results_df = pd.DataFrame.from_dict(
        results, orient="index", columns=["Predicted", "Actual"]
    )
    results_csv_path = os.path.join(outputs_path, "results.csv")
    results_df.to_csv(results_csv_path)

    summary_path = os.path.join(outputs_path, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.2f}%\n\n")
        f.write("Differences:\n")
        for diff in differences:
            f.write(
                f"{diff['filename']}: Predicted '{diff['predicted']}', Actual '{diff['actual']}' (Distance: {diff['distance']})\n"
            )

    print(f"Test completed. Results saved to {results_csv_path} and {summary_path}")


if __name__ == "__main__":
    main()
