import kagglehub
import shutil
from pathlib import Path
from src.config import Config


def main():
    dataset_path = kagglehub.dataset_download(
        "piotrstefaskiue/poland-vehicle-license-plate-dataset"
    )
    config = Config()

    src_photos = Path(dataset_path) / "photos"
    src_annotations = Path(dataset_path) / "annotations.xml"

    print("Copying photos...")
    for img_file in src_photos.glob("*.jpg"):
        shutil.copy(img_file, config.RAW_IMAGES_PATH)

    print("Copying annotations...")
    shutil.copy(src_annotations, config.DATA_PATH)

    print(f"Dataset copied to: {config.RAW_IMAGES_PATH}")
    print(f"Annotations copied to: {config.ANNOTATIONS_PATH}")


if __name__ == "__main__":
    main()
