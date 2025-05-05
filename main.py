import os
from tqdm import tqdm
import pandas as pd
from src.config import Config
from src.data_loader import parse_annotations
from src.evaluation import evaluate_results
from src.plate_detection import predict_plate_number
import cv2


def main():
    config = Config()

    annotations = parse_annotations(config.ANNOTATIONS_PATH)

    results = {}
    for filename, actual_plate in tqdm(annotations.items()):
        img_path = os.path.join(config.RAW_IMAGES_PATH, filename)
        if not os.path.exists(img_path):
            continue
        predicted_plate, processed_image = predict_plate_number(img_path)
        if processed_image is not None:
            processed_image_path = os.path.join(
                config.PROCESSED_IMAGES_PATH, "processed_" + filename
            )
            cv2.imwrite(processed_image_path, processed_image)

        results[filename] = (predicted_plate, actual_plate)

    accuracy, differences = evaluate_results(results)

    results_df = pd.DataFrame.from_dict(
        results, orient="index", columns=["Predicted", "Actual"]
    )
    results_df.to_csv(config.RESULTS_PATH)

    with open(config.SUMMARY_PATH, "w") as f:
        f.write(f"Accuracy: {accuracy:.2f}%\n\n")
        f.write("Differences:\n")
        for diff in differences:
            f.write(
                f"{diff['filename']}: Predicted '{diff['predicted']}', Actual '{diff['actual']}' (Distance: {diff['distance']})\n"
            )


if __name__ == "__main__":
    main()
