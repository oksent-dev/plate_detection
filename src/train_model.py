import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
import torch
import cv2
import re
from sklearn.model_selection import train_test_split
from config import Config
from bs4 import BeautifulSoup
from ultralytics import YOLO


def parse_annotations(annot_folder: str) -> pd.DataFrame:
    """
    Parse the XML annotations and convert them to a DataFrame.

    Parameters:
    - annot_folder: Path to the folder containing XML annotations.

    Returns:
    - DataFrame containing image paths and bounding box coordinates.
    """
    xml_files = sorted(
        os.listdir(annot_folder), key=lambda x: int(re.search(r"\d+", x).group())
    )
    img_labels = []
    for file in xml_files:
        with open(os.path.join(annot_folder, file), "r") as w:
            xml = w.read()
            soup = BeautifulSoup(xml, "xml")
            xmin = int(soup.find(["xmin"]).get_text())
            ymin = int(soup.find(["ymin"]).get_text())
            xmax = int(soup.find(["xmax"]).get_text())
            ymax = int(soup.find(["ymax"]).get_text())
            width = int(soup.find(["width"]).get_text())
            height = int(soup.find(["height"]).get_text())
            name = soup.find(["name"]).get_text()
            filename = soup.find(["filename"]).get_text()

            data = {
                "img_path": filename.split(".")[0],
                "img_width": width,
                "img_height": height,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
            img_labels.append(data)
    df = pd.DataFrame(img_labels)
    df["x_cent"] = (df["xmin"] + df["xmax"]) / 2 / df["img_width"]
    df["y_cent"] = (df["ymin"] + df["ymax"]) / 2 / df["img_height"]
    df["width"] = (df["xmax"] - df["xmin"]) / df["img_width"]
    df["height"] = (df["ymax"] - df["ymin"]) / df["img_height"]
    return df


def convert_data_yolo_format(data, folder_type, img_folder):
    config = Config()
    img_path = (
        config.DATA_PATH
        / "training_dataset"
        / "car_license_plate"
        / folder_type
        / "images"
    )
    labels_path = (
        config.DATA_PATH
        / "training_dataset"
        / "car_license_plate"
        / folder_type
        / "labels"
    )

    os.makedirs(img_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    for indx, row in data[
        ["img_path", "x_cent", "y_cent", "width", "height"]
    ].iterrows():
        text_write = f"0 {row['x_cent']:.4f} {row['y_cent']:.4f} {row['width']:.4f} {row['height']:.4f}\n"
        with open(os.path.join(labels_path, row["img_path"] + ".txt"), "w+") as file:
            file.write(text_write)

        shutil.copy(
            os.path.join(img_folder, row["img_path"] + ".png"),
            os.path.join(img_path, row["img_path"]) + ".png",
        )

    print(f"YOLO format conversion for {folder_type} set completed!")


def create_yaml_file() -> None:
    """
    Create a YAML file for the YOLOv8 dataset configuration.
    """
    config = Config()
    yaml_data = f"""
    path: {config.DATA_PATH / "training_dataset" / "car_license_plate"}
    train: {config.DATA_PATH / "training_dataset" / "car_license_plate" / "train" / "images"}
    val: {config.DATA_PATH / "training_dataset" / "car_license_plate" / "val" / "images"}
    test: {config.DATA_PATH / "training_dataset" / "car_license_plate" / "test" / "images"}
    nc: 1
    names: ["license_plate"]
    """

    datasets_yaml_path = config.DATA_PATH / "datasets.yaml"
    with open(datasets_yaml_path, "w") as file:
        file.write(yaml_data)
    print(f"datasets.yaml created at {datasets_yaml_path}")


def train_model() -> None:
    """
    Train the YOLOv8 model using the training dataset.
    """
    config = Config()
    print(f"CuDA Available: {torch.cuda.is_available()}")

    img_folder = config.DATA_PATH / "training"
    annot_folder = config.DATA_PATH / "training_annotations"

    img_folder.mkdir(exist_ok=True, parents=True)
    annot_folder.mkdir(exist_ok=True, parents=True)

    df = parse_annotations(annot_folder)

    train, test = train_test_split(df, test_size=1 / 10, random_state=42)
    train, val = train_test_split(train, train_size=8 / 9, random_state=42)

    print(f"TRAIN: {len(train)}, TEST: {len(test)}, VAL: {len(val)}")
    convert_data_yolo_format(train, "train", img_folder)
    convert_data_yolo_format(val, "val", img_folder)
    convert_data_yolo_format(test, "test", img_folder)

    create_yaml_file()

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(config.DATA_PATH / "datasets.yaml"),
        epochs=100,
        batch=16,
        imgsz=320,
        cache=True,
        device=0,
    )


def main():
    config = Config()
    if not os.path.exists("runs/detect/train/weights/best.pt"):
        print("Training the model...")
        train_model()
    model = YOLO("runs/detect/train/weights/best.pt")
    print("Model loaded successfully!")
    image_path = config.DATA_PATH / "raw" / "106.jpg"
    results = model.predict(image_path)
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_score = box.conf[0]
            cls = box.cls[0]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"SCORE: {conf_score*100:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )

    plt.imshow(image)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
