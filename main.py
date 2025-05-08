from src.config import config
from src.data_loader import parse_annotations, load_images
from src.evaluation import evaluate_results, create_summary, create_results_csv
from src.process_images import process_images


def main():
    annotations = parse_annotations(config.ANNOTATIONS_PATH)
    images = load_images(annotations)

    results, total_time = process_images(images, annotations)

    create_results_csv(results)

    correct, total, accuracy, differences, time_per_100_images, final_grade = (
        evaluate_results(results, total_time)
    )

    create_summary(
        correct,
        total,
        accuracy,
        time_per_100_images,
        final_grade,
        differences,
    )


if __name__ == "__main__":
    main()
