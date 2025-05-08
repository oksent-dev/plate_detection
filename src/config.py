from pathlib import Path


class Config:
    SAVE_IMAGES = True
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_PATH = PROJECT_ROOT / "data"
    RAW_IMAGES_PATH = DATA_PATH / "raw"
    PROCESSED_IMAGES_PATH = DATA_PATH / "processed"

    OUTPUTS = PROJECT_ROOT / "outputs"
    ANNOTATIONS_PATH = DATA_PATH / "annotations.xml"

    RESULTS_PATH = OUTPUTS / "results.csv"
    SUMMARY_PATH = OUTPUTS / "summary.txt"

    OUTPUTS.mkdir(exist_ok=True, parents=True)
    DATA_PATH.mkdir(exist_ok=True, parents=True)
    RAW_IMAGES_PATH.mkdir(exist_ok=True, parents=True)
    PROCESSED_IMAGES_PATH.mkdir(exist_ok=True, parents=True)


config = Config()
