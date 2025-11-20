from pathlib import Path

import cv2
import kagglehub
import numpy as np
from cv2.typing import MatLike


def download_dataset(dataset: str):
    """Download the dataset from Kaggle"""
    print("Downloading dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download(dataset)
    print(f"Dataset downloaded to: {dataset_path}")
    return dataset_path


def load_images_and_labels(
    dataset_path: str, image_size: tuple[int, int]
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], set[str]]:
    """Load images and labels from the dataset directory"""

    X_train: list[MatLike] = []
    X_valid: list[MatLike] = []

    y_train: list[str] = []
    y_valid: list[str] = []

    train_root = Path(f"{dataset_path}/train")
    valid_root = Path(f"{dataset_path}/test")

    # Get paths for all train directories
    train_class_dirs: list[Path] = []
    for d in train_root.iterdir():
        train_class_dirs.append(d)

    # Get paths for all validation directories
    valid_class_dirs: list[Path] = []
    for d in valid_root.iterdir():
        valid_class_dirs.append(d)

    print(f"{len(train_class_dirs)} image classes:")
    for class_dir in train_class_dirs:
        print(f"  - {class_dir.name}")

    # Load images from each class for train
    for class_dir in train_class_dirs:
        class_name = class_dir.name
        print(f"\nLoading images from class: {class_name}")

        # Get all image files
        image_files = list(class_dir.glob("*.jpg"))

        for img_path in image_files:
            try:
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Resize
                img = cv2.resize(img, image_size)

                X_train.append(img)
                y_train.append(class_name)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        print(
            f"  Loaded {len([label for label in y_train if label == class_name])} images"
        )

    # Load images from each class for train
    for class_dir in valid_class_dirs:
        class_name = class_dir.name
        print(f"\nLoading images from class: {class_name}")

        # Get all image files
        image_files = list(class_dir.glob("*.jpg"))

        for img_path in image_files:
            try:
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Resize
                img = cv2.resize(img, image_size)

                X_valid.append(img)
                y_valid.append(class_name)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

        print(
            f"  Loaded {len([label for label in y_valid if label == class_name])} images"
        )

    classes = set(y_train)
    print(f"\nTotal images loaded: train: {len(X_train)}, valid: {len(X_valid)}")
    print(f"Classes: {classes}")

    train_dataset = (np.array(X_train), np.array(y_train))
    valid_dataset = (np.array(X_valid), np.array(y_valid))

    return (train_dataset, valid_dataset, classes)
