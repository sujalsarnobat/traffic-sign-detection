"""
Inference module for traffic sign detection.
Supports: single image, video file, real-time webcam.

Usage:
    python src/predict.py --mode image   --input path/to/image.jpg  --model models/traffic_sign_model.keras
    python src/predict.py --mode video   --input path/to/video.mp4  --model models/traffic_sign_model.keras
    python src/predict.py --mode webcam                              --model models/traffic_sign_model.keras
"""

import argparse
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

from class_names import CLASS_NAMES
from data_loader import preprocess_single_image

# Colours for overlay (BGR)
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 0)
BG_COLOR = (0, 255, 0)
WARN_COLOR = (0, 0, 255)

CONF_THRESHOLD = 0.60  # below this → show warning


def load_model(model_path: str) -> tf.keras.Model:
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Train first: python src/train.py --data_dir data/"
        )
    print(f"Loading model from {model_path} ...")
    return tf.keras.models.load_model(model_path)


def predict_image(model: tf.keras.Model, img: np.ndarray) -> tuple[int, float, str]:
    """
    Run inference on a single BGR OpenCV image.
    Returns (class_id, confidence, label)
    """
    tensor = preprocess_single_image(img)
    preds = model.predict(tensor, verbose=0)[0]
    class_id = int(np.argmax(preds))
    confidence = float(preds[class_id])
    label = CLASS_NAMES.get(class_id, "Unknown")
    return class_id, confidence, label


def draw_overlay(frame: np.ndarray, label: str, confidence: float, region=None) -> np.ndarray:
    """Draw prediction overlay on frame."""
    h, w = frame.shape[:2]
    color = BG_COLOR if confidence >= CONF_THRESHOLD else WARN_COLOR
    status = f"{label}  {confidence*100:.1f}%"

    if region is not None:
        x1, y1, x2, y2 = region
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Top banner
    cv2.rectangle(frame, (0, 0), (w, 52), color, -1)
    cv2.putText(frame, status, (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, 2, cv2.LINE_AA)

    if confidence < CONF_THRESHOLD:
        cv2.putText(frame, "LOW CONFIDENCE", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WARN_COLOR, 2, cv2.LINE_AA)
    return frame


# ─── Image mode ───────────────────────────────────────────────────────────────

def run_image(model_path: str, image_path: str, output_path: str = None):
    model = load_model(model_path)
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    class_id, confidence, label = predict_image(model, frame)
    print(f"\nPrediction : {label}")
    print(f"Confidence : {confidence*100:.2f}%")
    print(f"Class ID   : {class_id}")

    result = draw_overlay(frame.copy(), label, confidence)

    if output_path:
        cv2.imwrite(output_path, result)
        print(f"Saved result → {output_path}")
    else:
        cv2.imshow("Traffic Sign Detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ─── Video / Webcam mode ───────────────────────────────────────────────────────

def run_video_or_webcam(model_path: str, source, output_path: str = None):
    model = load_model(model_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Running inference... Press 'q' to quit.")
    frame_count = 0
    skip_frames = 2  # predict every N frames for speed

    last_label, last_conf = "Initialising...", 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:
            _, last_conf, last_label = predict_image(model, frame)

        result = draw_overlay(frame.copy(), last_label, last_conf)
        frame_count += 1

        if writer:
            writer.write(result)

        cv2.imshow("Traffic Sign Detection — press Q to quit", result)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
        print(f"Saved output → {output_path}")
    cv2.destroyAllWindows()


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Sign Inference")
    parser.add_argument("--mode", choices=["image", "video", "webcam"], required=True)
    parser.add_argument("--input", type=str, default=None,
                        help="Path to image or video (not needed for webcam)")
    parser.add_argument("--model", type=str, default="models/traffic_sign_model.keras")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional: save result to this path")
    args = parser.parse_args()

    if args.mode == "image":
        if not args.input:
            parser.error("--input required for image mode")
        run_image(args.model, args.input, args.output)

    elif args.mode == "video":
        if not args.input:
            parser.error("--input required for video mode")
        run_video_or_webcam(args.model, args.input, args.output)

    elif args.mode == "webcam":
        run_video_or_webcam(args.model, 0, args.output)
