# 🎙️ Audio Emotion Recognition System (AERS)

A deep learning project that recognizes human emotions from speech audio using **Mel Spectrograms** and a **Convolutional Neural Network (CNN)**. Trained on RAVDESS and evaluated cross-corpus on TESS to test real-world generalization.

---

## 📌 Overview

This system processes raw `.wav` audio files, converts them into Mel Spectrogram images, and feeds them into a CNN to classify speech into **7 emotion categories**:

`Neutral` · `Happy` · `Sad` · `Angry` · `Fearful` · `Disgust` · `Surprised`

The key challenge this project addresses is **cross-corpus generalization** — training on one dataset (RAVDESS) and testing on a completely different one (TESS) to evaluate how well the model generalizes beyond the speakers it was trained on.

---

## 📂 Datasets

| Dataset | Speakers | Emotions | Files | Role |
|---------|----------|----------|-------|------|
| [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) | 24 actors (12M / 12F) | 7 (calm dropped) | ~840 | Training |
| [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) | 2 actresses | 7 | ~2800 | Cross-corpus Testing |

> **Note:** Only audio-only speech files from RAVDESS are used (modality=03, channel=01). The "calm" emotion is dropped since TESS has no equivalent label.

---

## 🏗️ Project Pipeline

```
Raw .wav
  └─ Load mono @ 22050 Hz, pad/truncate to 3s
      └─ Mel Spectrogram (128 mel bins)
          └─ Convert to dB scale (power_to_db)
              └─ Normalize to [0, 1]
                  └─ Resize to 128×128
                      └─ Shape: (128, 128, 1) → CNN Input
```

---

## 🧠 Model Architecture

```
Input (128, 128, 1)
│
├── Conv2D(32) + BatchNorm + ReLU + MaxPool(2×2) + Dropout(0.3)
├── Conv2D(64) + BatchNorm + ReLU + MaxPool(2×2) + Dropout(0.3)
├── Conv2D(128) + BatchNorm + ReLU + MaxPool(2×2) + Dropout(0.4)
│
├── GlobalAveragePooling2D
├── Dense(128) + ReLU + Dropout(0.5)
└── Dense(7, softmax)
```

---

## ✨ Key Features

- **Mel Spectrogram extraction** using `librosa` — treats audio as a 2D image for the CNN
- **Data augmentation** (time stretching + noise injection) applied only to RAVDESS training data to combat the small dataset size (~840 files)
- **Class weight balancing** via `sklearn` to prevent the model from biasing toward majority classes
- **Cross-corpus evaluation** — model trained on RAVDESS, tested on TESS with no retraining

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `librosa` | Audio loading & Mel Spectrogram extraction |
| `Pillow` | Spectrogram image resizing |
| `TensorFlow / Keras` | CNN model building & training |
| `scikit-learn` | Train/test split & class weight computation |
| `NumPy / Pandas` | Data manipulation |
| `Matplotlib / Seaborn` | Training history & confusion matrix plots |

---

## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/Harpil70/Audio-Emotion-Recognition-System.git
cd Audio-Emotion-Recognition-System
```

**2. Download the datasets** from the links above and place them inside a `dataset/` folder:
```
dataset/
├── RAVDESS/
│   ├── Actor_01/
│   ├── Actor_02/
│   └── ...
└── TESS/
    ├── OAF_angry/
    ├── YAF_happy/
    └── ...
```

**3. Install dependencies**
```bash
pip install pandas numpy matplotlib librosa pillow scikit-learn tensorflow
```

**4. Run the notebook**
```bash
jupyter notebook AERS.ipynb
```

---

## 📊 Results

| Evaluation | Accuracy |
|------------|----------|
| RAVDESS (validation) | ~75–93% |
| TESS (cross-corpus) | ~45–52% |

The accuracy drop between RAVDESS and TESS reflects **domain shift** — differences in speaker age, gender distribution, and recording conditions between the two datasets. This is a known challenge in cross-corpus speech emotion recognition.

---

## 📁 Project Structure

```
Audio-Emotion-Recognition-System/
├── AERS.ipynb          ← Main notebook
├── dataset/
│   ├── RAVDESS/
│   └── TESS/
└── README.md
```
