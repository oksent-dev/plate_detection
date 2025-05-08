import kagglehub
import shutil
from pathlib import Path
from src.config import Config


def main():
    dataset_path = kagglehub.dataset_download("andrewmvd/car-plate-detection")
    config = Config()
    print("path to dataset", dataset_path)
    src_photos = Path(dataset_path) / "images"
    src_annotations = Path(dataset_path) / "annotations"

    print("Copying photos...")
    for img_file in src_photos.glob("*.png"):
        shutil.copy(img_file, config.TRAINING_IMAGES_PATH)

    print("Copying annotations...")
    for xml_file in src_annotations.glob("*.xml"):
        shutil.copy(xml_file, config.TRAINING_ANNOTATIONS_PATH)

    print(f"Dataset copied to: {config.TRAINING_IMAGES_PATH}")
    print(f"Annotations copied to: {config.TRAINING_ANNOTATIONS_PATH}")


if __name__ == "__main__":
    main()
