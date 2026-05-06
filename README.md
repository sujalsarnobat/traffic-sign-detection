# 🚦 Traffic Sign Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A deep learning system that classifies 43 types of traffic signs from images, video files, and real-time webcam streams.**

[Features](#-features) · [Demo](#-demo) · [Setup](#-setup) · [Usage](#-usage) · [Architecture](#-model-architecture) · [Results](#-results)

</div>

---

## 📌 Overview

This project implements a **Convolutional Neural Network (CNN)** trained on the [GTSRB (German Traffic Sign Recognition Benchmark)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) dataset — one of the most widely used benchmarks in autonomous driving research.

The system supports three inference modes:
- **Static image** classification with confidence scores
- **Video file** processing with frame-by-frame detection
- **Real-time webcam** detection at 30+ FPS

---

## ✨ Features

| Feature | Details |
|---|---|
| 🧠 Custom CNN | 3 conv blocks + BatchNorm + Dropout, ~95% val accuracy |
| 📸 Image detection | Upload any image, get top-3 predictions with confidence |
| 🎥 Webcam stream | Real-time detection via OpenCV + Streamlit |
| 📊 EDA Notebook | Full dataset exploration & training walkthrough |
| 🧪 Unit tests | pytest suite covering model, preprocessing, class labels |
| 🌐 Web UI | Streamlit app with Plotly confidence chart |
| 📈 Training plots | Accuracy/loss curves + confusion matrix auto-saved |

---

## 🎬 Demo

> 📸 *Add screenshots here after running the app*

<!-- 
  Suggested screenshots to add to assets/ folder:
  - assets/demo_image.png   → image upload prediction
  - assets/demo_webcam.png  → webcam detection frame
  - assets/training_curves.png → copy from models/ after training
-->

| Image Upload | Webcam Detection |
|:---:|:---:|
| ![Image demo](assets/demo_image.png) | ![Webcam demo](assets/demo_webcam.png) |

---

## 🗂 Project Structure

```
traffic-sign-detection/
│
├── src/
│   ├── class_names.py       # All 43 GTSRB class labels
│   ├── data_loader.py       # Data pipeline: load, preprocess, augment
│   ├── model.py             # CNN architecture definition
│   ├── train.py             # Training script with callbacks
│   └── predict.py           # Inference: image / video / webcam
│
├── app/
│   └── streamlit_app.py     # Web UI
│
├── notebooks/
│   └── exploration.ipynb    # EDA + training walkthrough
│
├── models/                  # Saved model weights (after training)
├── data/                    # Dataset goes here (see Setup)
├── tests/
│   └── test_core.py         # pytest unit tests
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone & install

```bash
git clone https://github.com/sujalsarnobat/traffic-sign-detection.git
cd traffic-sign-detection
pip install -r requirements.txt
```

### 2. Download the dataset

Download the **GTSRB dataset** from Kaggle:

👉 [https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

Extract it so the structure looks like:

```
data/
└── Train/
    ├── 0/        ← Speed limit 20km/h
    ├── 1/        ← Speed limit 30km/h
    ├── ...
    └── 42/
```

> The `data/` folder is intentionally excluded from Git (too large). The `.gitkeep` file marks it as the intended location.

---

## 🚀 Usage

### Train the model

```bash
python src/train.py --data_dir data/ --epochs 30
```

Optional flags:
```
--epochs       Number of training epochs (default: 30)
--batch_size   Batch size (default: 64)
--lr           Learning rate (default: 0.001)
--output_dir   Where to save model & plots (default: models/)
```

---

### Run inference

**On an image:**
```bash
python src/predict.py --mode image --input path/to/sign.jpg --model models/traffic_sign_model.keras
```

**On a video:**
```bash
python src/predict.py --mode video --input path/to/video.mp4 --model models/traffic_sign_model.keras
```

**Real-time webcam:**
```bash
python src/predict.py --mode webcam --model models/traffic_sign_model.keras
```

---

### Launch the web app

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

### Run tests

```bash
pytest tests/ -v
```

---

## 🧠 Model Architecture

```
Input (32×32×3)
    │
    ├─ Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU → MaxPool → Dropout(0.25)
    │
    ├─ Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.25)
    │
    ├─ Conv2D(128) → BN → ReLU → MaxPool → Dropout(0.25)
    │
    ├─ Flatten → Dense(512) → BN → ReLU → Dropout(0.5)
    │
    └─ Dense(43, softmax)
```

- **L2 regularization** on all Conv/Dense layers
- **BatchNormalization** for faster, stable training
- **Data augmentation**: rotation, zoom, translation, brightness, contrast

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | ~95%+ |
| Dataset | GTSRB (39,209 training images) |
| Classes | 43 |
| Input size | 32×32×3 |

> Training curves and confusion matrix are auto-saved to `models/` after training.

---

## 🛠 Tech Stack

- **TensorFlow / Keras** — model training & inference
- **OpenCV** — image/video processing, webcam capture
- **Streamlit** — web UI
- **Plotly** — interactive confidence charts
- **scikit-learn** — train/val split, evaluation metrics
- **pytest** — unit testing

---

## 📄 License

MIT License — free to use and modify.

---

<div align="center">
Built by <a href="https://github.com/sujalsarnobat">Sujal Sarnobat</a>
</div>
