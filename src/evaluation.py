import pandas as pd
from src.config import config
from Levenshtein import distance as levenshtein_distance


def evaluate_results(results: dict, total_time: float) -> tuple:
    """
    Evaluates the results of the OCR predictions against the actual license plates.

    Parameters:
    - results: Dictionary containing the results of predictions
    - total_time: Total time taken to process the images

    Returns:
    - correct: Number of correctly predicted plates
    - total: Total number of plates
    - accuracy: Accuracy percentage
    - differences: List of differences between predicted and actual plates
    - time_per_100_images: Processing time per 100 images in seconds
    - final_grade: Final grade based on accuracy and processing time
    """
    total = len(results)
    correct = 0
    differences = []

    for filename, (predicted, actual) in results.items():
        if predicted == actual:
            correct += 1
        else:
            diff = {
                "filename": filename,
                "predicted": predicted,
                "actual": actual,
                "distance": levenshtein_distance(predicted, actual),
            }
            differences.append(diff)

    accuracy = (correct / total) * 100 if total > 0 else 0
    time_per_100_images = (total_time / total) * 100
    final_grade = calculate_final_grade(accuracy, time_per_100_images)

    return correct, total, accuracy, differences, time_per_100_images, final_grade


def calculate_final_grade(accuracy_percent: float, processing_time_sec: float) -> str:
    """
    Calculates the final grade based on license plate OCR accuracy and pro
    cessing time.

    Parameters:
    - accuracy_percent: OCR accuracy as a percentage (0-100)
    - processing_time_sec: total time to process 100 images in seconds

    Returns:
    - Grade on a scale from 2.0 to 5.0 (rounded to the nearest 0.5)
    """
    if accuracy_percent < 60 or processing_time_sec > 60:
        return 2.0
    accuracy_norm = (accuracy_percent - 60) / 40
    time_norm = (60 - processing_time_sec) / 50
    score = 0.7 * accuracy_norm + 0.3 * time_norm
    grade = 2.0 + 3.0 * score

    return round(grade * 2) / 2


def create_summary(
    correct: int,
    total: int,
    accuracy: float,
    time_per_100_images: float,
    final_grade: float,
    differences: list,
) -> None:
    """
    Creates a summary of the evaluation results and writes it to a file.

    Parameters:
    - correct: Number of correctly predicted plates
    - total: Total number of plates
    - accuracy: Accuracy percentage
    - time_per_100_images: Processing time per 100 images in seconds
    - final_grade: Final grade based on accuracy and processing time
    - differences: List of differences between predicted and actual plates
    """
    with open(config.SUMMARY_PATH, "w") as f:
        f.write(f"Correctly predicted: {correct} out of {total}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Processing time per 100 images: {time_per_100_images:.2f} seconds\n")
        f.write(f"Final grade: {final_grade:.1f}\n\n")
        f.write("Differences:\n")
        for diff in differences:
            f.write(
                f"{diff['filename']}: Predicted '{diff['predicted']}', Actual '{diff['actual']}' (Distance: {diff['distance']})\n"
            )


def create_results_csv(results: dict) -> None:
    """
    Creates a CSV file with the results of predictions.

    Parameters:
    - results: Dictionary containing the results
    """
    results_df = pd.DataFrame.from_dict(
        results, orient="index", columns=["Predicted", "Actual"]
    )
    results_df.to_csv(config.RESULTS_PATH, index_label="Filename")
