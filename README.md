# Autonomous Vehicle — Behavioral Cloning

> **ML Project** — Training a car to drive itself using only camera images.

---

## Project Overview

This project implements **Behavioral Cloning** for autonomous driving. A Convolutional Neural Network (CNN) is trained to map images from a dashcam directly to steering angles — learning to drive by imitating human driving data.

Instead of complex reinforcement learning, we use **supervised learning**:
- **Input:** A camera image from the road
- **Output:** A predicted steering angle (continuous value)

The model is trained on real driving data from the **Udacity Self-Driving Car Dataset** and tested in the **Udacity Unity Simulator**.

---

## Model Architecture — NVIDIA End-to-End CNN

We implement the architecture from NVIDIA's research paper:
> *"End to End Learning for Self-Driving Cars"* — Bojarski et al., 2016

The network is built **entirely from scratch** using TensorFlow/Keras:

```
Input Image (66×200×3 — YUV colorspace)
    ↓
Conv2D(24, 5×5, stride=2, ELU)
Conv2D(36, 5×5, stride=2, ELU)
Conv2D(48, 5×5, stride=2, ELU)
Conv2D(64, 3×3, ELU)
Conv2D(64, 3×3, ELU)
    ↓
Dropout(0.5)
Flatten
Dense(100, ELU)
Dense(50,  ELU)
Dense(10,  ELU)
Dense(1)   ← Predicted Steering Angle
```

---

## Project Structure

```
autonomous-vehicle-ml/
│
├── drive.py                               # Main file — run this to drive the car
├── Behavioral_Cloning_Colab_Final.ipynb   # Training notebook (run on Google Colab)
├── requirements.txt                       # Python dependencies
├── README.md
├── .gitignore
│
├── support/
│   └── download_data.py                   # Helper script to download the dataset
│
├── data/                                  # Udacity driving dataset
│   ├── driving_log.csv                    # Steering angles for each frame
│   └── IMG/                              # Camera images (center, left, right)
│
└── others/
    └── autonomus record.mp4               # Demo recording of autonomous driving
```

---

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get the Dataset
The dataset is included in the `data/` folder. Alternatively, download it fresh:
```bash
python support/download_data.py
```

### 3. Train the Model
Open **`Behavioral_Cloning_Colab_Final.ipynb`** in Google Colab for GPU acceleration:
- Upload the notebook to [Google Colab](https://colab.research.google.com/)
- Upload your `data/` folder or mount Google Drive
- Run all cells → the best model is saved as `best_model.h5`

### 4. Run Autonomous Mode
1. Open the **Udacity Simulator** (`Default Windows desktop 64-bit.exe`)
2. Choose a track → click **"Autonomous Mode"**
3. In a terminal, run:
```bash
python drive.py best_model.h5
```
4. Watch the car drive itself!

---

## Training Pipeline

| Step | Description |
|---|---|
| **Data Loading** | Parse `driving_log.csv`, use all 3 cameras (center, left, right) |
| **Augmentation** | Random flip, brightness, panning — prevents overfitting |
| **Generator** | Loads images in batches — memory efficient |
| **Training** | Adam optimizer, MSE loss, EarlyStopping + ModelCheckpoint |
| **Evaluation** | Loss curves, True vs Predicted scatter plot |

---

## Key Concepts

- **Behavioral Cloning** — Learning to drive by imitating human demonstrations
- **Data Augmentation** — Artificially expanding training data with transformations
- **CNN (Convolutional Neural Network)** — Extracts road features from images
- **Regression** — Predicting a continuous value (steering angle), not a category

---

## Demo

A recorded demo of the car driving autonomously is available in `others/autonomus record.mp4`.

---

## Team

University ML Project — Autonomous Vehicle using Behavioral Cloning

---

## References

- NVIDIA Paper: [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)
- Udacity Simulator: [github.com/udacity/self-driving-car-sim](https://github.com/udacity/self-driving-car-sim)
- Dataset: [Udacity Self-Driving Car Dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip)
