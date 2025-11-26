
<h1 align="center">RealWaste CNN – Image Classification</h1>

<p align="center">
  <a href="https://github.com/RealWaste-CNN/RealWaste_CNN_Banuka/stargazers">
    <img src="https://img.shields.io/github/stars/RealWaste-CNN/RealWaste_CNN_Banuka?style=for-the-badge" alt="GitHub stars" />
  </a>
  <a href="https://github.com/RealWaste-CNN/RealWaste_CNN_Banuka/network/members">
    <img src="https://img.shields.io/github/forks/RealWaste-CNN/RealWaste_CNN_Banuka?style=for-the-badge" alt="GitHub forks" />
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/github/license/RealWaste-CNN/RealWaste_CNN_Banuka?style=for-the-badge" alt="License: MIT" />
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/deep%20learning-CNN-blueviolet?style=flat-square" />
  <img src="https://img.shields.io/badge/status-research--project-informational?style=flat-square" />
</p>

<!-- Animated typing banner (hosted service) -->
<p align="center">
  <a href="https://github.com/RealWaste-CNN/RealWaste_CNN_Banuka">
    <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=20&pause=1200&center=true&vCenter=true&width=650&lines=Real-world+waste+image+classification+with+CNNs;Deep+learning+for+smart+recycling+and+sorting;From+raw+images+to+actionable+insights" alt="RealWaste CNN Typing Animation" />
  </a>
</p>

---

## 📌 Overview

**RealWaste CNN** is a deep-learning project for **classifying real-world waste images** into predefined categories (e.g., plastics, metal, paper, etc.).  
It is designed as an **experimental baseline** and teaching/benchmarking resource for:

- Waste-sorting and recycling applications  
- Smart bins and robotics platforms  
- Research on computer vision for sustainability

The repository includes:

- Data organization for the **RealWaste-like dataset**  
- CNN training and evaluation pipeline  
- Scripts for experiments and result logging  
- A report summarizing methodology and key findings

---

## 🎬 Visual Preview



<p align="center">
  <img src="assets/realwaste-demo.gif" alt="RealWaste CNN Demo" width="650" />
</p>

<p align="center">
  <em>Example: animated demo of the model classifying incoming waste images.</em>
</p>

---

## 🧠 Key Features

- **End-to-end CNN pipeline** for waste image classification  
- **Configurable training** (hyperparameters, splits, augmentations)  
- **Repeatable experiments** with saved logs and results in `results/`  
- Clear separation between **data, source code, results, and report**  
- Designed to be **extended** with new architectures or datasets

---

## 🗂️ Project Structure


RealWaste_CNN_Banuka/
├── data/            # Dataset structure (train/val/test splits, metadata)
├── report/          # Project report, figures, and documentation
├── results/         # Saved models, logs, metrics, confusion matrices, etc.
├── src/             # Source code (data loading, models, training, evaluation)
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
## 🚀 Getting Started
1. Clone the repository
git clone https://github.com/RealWaste-CNN/RealWaste_CNN_Banuka.git
cd RealWaste_CNN_Banuka

2. Create and activate a virtual environment (recommended)
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

## 📊 Dataset

The project assumes a RealWaste-style dataset organized roughly as:

data/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── val/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...


Typical preprocessing & augmentations include:

Resizing and center-cropping to a fixed input size

Normalization based on dataset statistics

Random horizontal flips, rotations, and/or color jitter (configurable)

Please check the dataset loading code in src/ for the exact expected directory structure and transforms.

## ⚙️ Training

Update script names and arguments below to match your implementation in src/.

Basic training run
python src/train.py \
  --data_dir data \
  --output_dir results/exp01 \
  --epochs 30 \
  --batch_size 64 \
  --lr 1e-3


Typical options (subject to your implementation):

--model – choose architecture (e.g., baseline_cnn, resnet18)

--augment – enable/disable data augmentation

--seed – set random seed for reproducibility

## 📈 Evaluation & Inference
Evaluate on validation / test set
python src/evaluate.py \
  --data_dir data/test \
  --checkpoint results/exp01/best_model.pth


Expected outputs (saved under results/):

Overall accuracy

Per-class precision/recall/F1

Confusion matrix plots

Run inference on a single image
python src/inference.py \
  --image_path path/to/image.jpg \
  --checkpoint results/exp01/best_model.pth


Inference scripts typically print the predicted class and probability and may optionally save a visualization overlay.

## 🔁 Model & Training Pipeline (Animated)

<p align="center"> <img src="assets/realwaste-pipeline.gif" alt="RealWaste CNN Training Pipeline" width="700" /> </p>

High-level steps:

Load dataset → split into train/val/test

Preprocess & augment images

Forward pass through CNN

Compute loss & backpropagate gradients

Update weights using optimizer

Log metrics & save checkpoints in results/

## 📑 Results (Summary)

A detailed analysis (with tables, plots, and discussion) is available in the report/ directory.
Typical reported metrics include:

Overall test accuracy

Per-class performance (precision/recall/F1)

Confusion matrix visualizations

Discussion of failure modes and class imbalance

You can paste key plots (e.g., training curves and confusion matrices) here as static or animated images for quick viewing.

## 🧰 Tech Stack

Language: Python 3.8+

Deep Learning: (e.g.) PyTorch / TensorFlow (see requirements.txt)

Data Handling & Utilities: NumPy, pandas, etc.

Visualization: Matplotlib / Seaborn / other plotting libraries

Refer to requirements.txt for the definitive list of frameworks and versions.

## 🧪 Reproducibility

To keep experiments reproducible:

Use a fixed --seed where supported

Log hyperparameters and configuration in results/exp*/config.json (or similar)

Commit the exact requirements.txt used for your runs

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.

Fork the repository

Create a new branch: git checkout -b feature/my-feature

Commit your changes: git commit -m "Add my feature"

Push to the branch: git push origin feature/my-feature

Open a Pull Request

Please keep code style and documentation consistent with the existing project.

## 📚 Citation

If you use this repository or build upon it in your work, please consider citing it (for example in a report or thesis):

Liyanage, B. (2025). RealWaste CNN: Real-world Waste Image Classification using Convolutional Neural Networks. GitHub repository: https://github.com/RealWaste-CNN/RealWaste_CNN_Banuka


(Adjust the citation format as needed for your venue.)

## 📄 License

This project is licensed under the MIT License.
