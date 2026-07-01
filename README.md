# Voice Activity Detection with Handcrafted Features and LBP Encoding

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Status](https://img.shields.io/badge/status-work_in_progress-orange)

## 📖 Introduction
This project provides an implementation of **Voice Activity Detection (VAD)** using handcrafted acoustic features (MFCC, LFCC, STFT, CQCC) combined with **Local Binary Patterns (LBP)** and **Compound Local Binary Patterns (CLBP)** encoding. It is the companion codebase for the paper: *"Voice Activity Detection with Handcrafted Features and LBP Encoding"*. 

The system is designed to robustly separate human speech from background noise by analyzing texture-like properties of 2D acoustic representations (spectrograms, MFCC maps) using LBP algorithms.

## 📂 Repository Structure

The repository is currently organized as follows (and is undergoing a major refactoring):

```text
Voice-Activity-Detect/
├── data/                       # Contains raw and processed audio datasets, as well as extracted features (.npy)
├── paper_reference/            # Original paper and literature references
├── src/
│   ├── feature_extraction/     # Scripts and notebooks for MFCC, LFCC, STFT, CQCC + LBP/CLBP extraction
│   ├── inference/              # Inference pipeline for predicting voice activity
│   ├── preprocessing/          # Audio combining, silence filtering, and dataset preparation
│   ├── train/                  # Model training (SVM, Logistic Regression, Ensembles)
│   └── visualization/          # Comparing features and performance plots
└── README.md                   # Project documentation
```

## 🚀 Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Voice-Activity-Detect.git
   cd Voice-Activity-Detect
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   Ensure you have the required libraries installed (a `requirements.txt` will be provided soon). Core dependencies include:
   ```bash
   pip install numpy scipy librosa scikit-learn scikit-image matplotlib jupyter
   ```

## 🛠️ Usage Pipeline (Current)

Currently, the project is heavily notebook-driven. To reproduce the experiments:

1. **Preprocessing:** Use notebooks in `src/preprocessing/` to clean audio and generate combined datasets.
2. **Feature Extraction:** Run the respective feature notebook (e.g., `src/feature_extraction/mfcc_clbp_lib.ipynb`) to convert `.wav` files into `.npy` feature matrices.
3. **Training & Evaluation:** Use `src/train/model.ipynb` to train SVM / Logistic Regression classifiers on the extracted features.
4. **Visualization:** Analyze histograms and ROC curves in `src/visualization/`.

---

## 🏗️ Refactoring Roadmap (Clear Code Direction)

We are actively restructuring this codebase from a research prototype (Jupyter Notebooks) into a modular, scalable Python package. The planned architecture is:

- [ ] **Data Pipeline (`src/data_prep/`)**: Modularize dataset downloading, silence removal, and chunking into reusable Python scripts.
- [ ] **Feature Engineering API (`src/features/`)**: 
  - Create a unified object-oriented interface for extracting acoustic features (`MFCC`, `STFT`, `LFCC`).
  - Implement a common `LBPEncoder` and `CLBPEncoder` class for 2D feature mapping.
  - Remove hardcoded absolute paths (e.g., `E:/PythonFile/...`) and use relative configuration paths.
- [ ] **Training Modules (`src/models/`)**: Convert training notebooks into `train.py` and `evaluate.py` scripts with CLI arguments (using `argparse` or `hydra`) for easy hyperparameter tuning.
- [ ] **Inference Pipeline (`src/inference/`)**: Implement a `predict.py` script to accept a raw `.wav` file, extract LBP features on-the-fly, and output VAD predictions.
- [ ] **Configuration Management**: Introduce `config.yaml` to manage sample rates, FFT window sizes, LBP parameters, and model hyperparameters centrally.

## 📄 Citation

If you find this code helpful in your research, please refer to the main paper located in `paper_reference/Voice_Activity_Detection_with_Handcrafted_Features_and_LBP_Encoding.pdf`.
