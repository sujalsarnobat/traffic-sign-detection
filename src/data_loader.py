import os
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf

IMG_SIZE = 32
BATCH_SIZE = 64


def preprocess(images: np.ndarray) -> np.ndarray:
    """Normalize images to [0, 1]. For testing compatibility."""
    return images.astype(np.float32) / 255.0


def _load_image_label_paths(data_dir: str) -> tuple[list, list]:
    """
    Collect image paths and labels without loading images into memory.
    Returns lists of (image_path, label) pairs for lazy loading.
    """
    image_paths = []
    labels = []
    data_path = Path(data_dir) / "Train"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Train folder not found at {data_path}. "
            "Download the dataset from https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
        )

    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    print(f"Found {len(class_dirs)} classes.")

    for class_dir in class_dirs:
        class_id = int(class_dir.name)
        for img_path in class_dir.glob("*.png"):
            image_paths.append(str(img_path))
            labels.append(class_id)

    return image_paths, labels


def _load_and_preprocess_image(image_path: str) -> tf.Tensor:
    """Load and preprocess a single image from disk."""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img


def get_augmentation_layer() -> tf.keras.Sequential:
    """Returns a Keras augmentation pipeline."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ], name="augmentation")


def get_datasets(
    data_dir: str,
    val_split: float = 0.2,
    batch_size: int = BATCH_SIZE,
) -> tuple:
    """
    Optimized pipeline: lazy load → split → augment → batch.
    Images are loaded from disk on-demand during training.
    Returns (train_ds, val_ds, X_val_array, y_val_array)
    """
    print("\n[Loading dataset paths...]")
    image_paths, labels = _load_image_label_paths(data_dir)
    
    # Split paths and labels
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        image_paths, labels, test_size=val_split, random_state=42, stratify=labels
    )
    
    print(f"  Training samples: {len(X_train_paths)}")
    print(f"  Validation samples: {len(X_val_paths)}")

    # Create training dataset with lazy loading
    augment = get_augmentation_layer()
    train_ds = tf.data.Dataset.from_tensor_slices((X_train_paths, y_train))
    train_ds = (
        train_ds
        .shuffle(buffer_size=len(X_train_paths), reshuffle_each_iteration=True)
        .map(
            lambda path, label: (_load_and_preprocess_image(path), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(batch_size)
        .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Create validation dataset (on-demand with prefetch for training)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val_paths, y_val))
    val_ds = (
        val_ds
        .map(
            lambda path, label: (_load_and_preprocess_image(path), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # For evaluation metrics, load validation images using parallel TensorFlow ops
    print("[Loading validation data for evaluation...]")
    val_ds_eval = (
        tf.data.Dataset.from_tensor_slices(X_val_paths)
        .map(_load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
    )
    
    X_val_array = np.concatenate([batch.numpy() for batch in val_ds_eval], axis=0)

    return train_ds, val_ds, X_val_array, np.array(y_val, dtype=np.int32)


def preprocess_single_image(img: np.ndarray) -> np.ndarray:
    """Preprocess a single BGR OpenCV image for inference."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)
