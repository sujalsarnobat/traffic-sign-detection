"""
Train the Traffic Sign CNN on GTSRB dataset.

Usage:
    python src/train.py --data_dir data/ --epochs 30 --output_dir models/
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from model import build_model, compile_model
from data_loader import get_datasets
from class_names import CLASS_NAMES, NUM_CLASSES


def get_callbacks(output_dir: str) -> list:
    os.makedirs(output_dir, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(output_dir, "training_log.csv")
        ),
    ]


def plot_history(history, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    print(f"Saved training curves → {path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_dir: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 18))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=list(CLASS_NAMES.values()),
        yticklabels=list(CLASS_NAMES.values()),
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    print(f"Saved confusion matrix → {path}")
    plt.close()


def train(args):
    print(f"\n{'='*50}")
    print("  Traffic Sign Detection — Training")
    print(f"{'='*50}\n")

    # GPU check
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs available: {len(gpus)}")

    print("\n[1/4] Loading data...")
    train_ds, val_ds, X_val, y_val = get_datasets(
        data_dir=args.data_dir,
        val_split=0.2,
        batch_size=args.batch_size,
    )

    print("\n[2/4] Building model...")
    model = build_model(num_classes=NUM_CLASSES)
    model = compile_model(model, learning_rate=args.lr)
    model.summary()

    print("\n[3/4] Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=get_callbacks(args.output_dir),
    )

    print("\n[4/4] Evaluating...")
    y_pred = np.argmax(model.predict(X_val), axis=1)

    report = classification_report(
        y_val, y_pred,
        target_names=list(CLASS_NAMES.values()),
        output_dict=True,
    )
    report_path = os.path.join(args.output_dir, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Classification report saved → {report_path}")
    print(f"\nFinal Val Accuracy: {report['accuracy']:.4f}")

    plot_history(history, args.output_dir)
    plot_confusion_matrix(y_val, y_pred, args.output_dir)

    # Save final model
    final_path = os.path.join(args.output_dir, "traffic_sign_model.keras")
    model.save(final_path)
    print(f"\nModel saved → {final_path}")
    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Traffic Sign CNN")
    parser.add_argument("--data_dir", type=str, default="data/",
                        help="Path to GTSRB dataset directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output_dir", type=str, default="models/")
    args = parser.parse_args()
    train(args)
