import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

DIR_SPEECH = r"G:/SP2026/AIL303m/split_data/SPEECH"
DIR_MIXED  = r"G:/SP2026/AIL303m/split_data/MIXING DATA/SP+MU"

n_mfcc = 13

# ===============================
# FEATURE EXTRACTION
# ===============================
def extract_features(directory):
    mfcc_all = []
    rms_all  = []
    zcr_all  = []

    for file in os.listdir(directory):
        if not file.endswith(".wav"):
            continue

        y, sr = librosa.load(os.path.join(directory, file), sr=None)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        rms  = librosa.feature.rms(y=y)
        zcr  = librosa.feature.zero_crossing_rate(y)

        mfcc_all.append(np.mean(mfcc, axis=1))
        rms_all.append(np.mean(rms))
        zcr_all.append(np.mean(zcr))

    return (
        np.array(mfcc_all),   # (N, n_mfcc)
        np.array(rms_all),    # (N,)
        np.array(zcr_all)     # (N,)
    )

# ===============================
# COMPUTE FEATURES
# ===============================
mfcc_clean, rms_clean, zcr_clean = extract_features(DIR_SPEECH)
mfcc_mixed, rms_mixed, zcr_mixed = extract_features(DIR_MIXED)

mfcc_clean_mean = np.mean(mfcc_clean, axis=0)
mfcc_mixed_mean = np.mean(mfcc_mixed, axis=0)
mfcc_diff = mfcc_mixed_mean - mfcc_clean_mean

# ===============================
# VISUALIZATION
# ===============================
plt.figure(figsize=(10, 10))

# --- MFCC Clean ---
plt.subplot(4, 1, 1)
plt.imshow(mfcc_clean_mean[:, np.newaxis], aspect="auto", cmap="viridis")
plt.title("Mean MFCC – Speech (Clean)")
plt.ylabel("MFCC Index")
plt.colorbar()

# --- MFCC Mixed ---
plt.subplot(4, 1, 2)
plt.imshow(mfcc_mixed_mean[:, np.newaxis], aspect="auto", cmap="viridis")
plt.title("Mean MFCC – Speech (After Mixing)")
plt.ylabel("MFCC Index")
plt.colorbar()

# --- MFCC Difference ---
plt.subplot(4, 1, 3)
plt.imshow(mfcc_diff[:, np.newaxis], aspect="auto", cmap="coolwarm")
plt.title("MFCC Difference (Mixed − Clean)")
plt.ylabel("MFCC Index")
plt.colorbar()

# --- RMS & ZCR ---
plt.subplot(4, 1, 4)
plt.bar(["RMS Clean", "RMS Mixed"], 
        [np.mean(rms_clean), np.mean(rms_mixed)],
        alpha=0.8)

plt.bar(["ZCR Clean", "ZCR Mixed"], 
        [np.mean(zcr_clean), np.mean(zcr_mixed)],
        alpha=0.8)

plt.title("Energy (RMS) and Zero-Crossing Rate (ZCR)")
plt.tight_layout()
plt.show()
