import numpy as np
import librosa

def extract_stft(
    signal,
    sr=16000,
    n_fft=512,
    hop_length=160,   # 10 ms
    win_length=400    # 25 ms
):

    stft = librosa.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann"
    )

    # magnitude spectrogram
    spec = np.abs(stft)

    # log scale
    log_spec = librosa.amplitude_to_db(spec)

    return log_spec.T   # shape: (frames, freq_bins)

import numpy as np

def clbp(image, P=8, R=1):
    """
    image : 2D array (MFCC / spectrogram)
    P : số neighbor
    R : radius
    return:
        CLBP_S, CLBP_M, CLBP_C
    """
    h, w = image.shape

    # padding
    pad = R
    img = np.pad(image, pad, mode='edge')

    # center
    center = img[pad:pad+h, pad:pad+w]

    # init
    CLBP_S = np.zeros((h, w), dtype=np.uint8)
    CLBP_M = np.zeros((h, w), dtype=np.uint8)

    # magnitude threshold
    diff_list = []

    for p in range(P):
        theta = 2 * np.pi * p / P
        dy = int(round(R * np.sin(theta)))
        dx = int(round(R * np.cos(theta)))

        neighbor = img[pad+dy:pad+dy+h, pad+dx:pad+dx+w]

        diff = neighbor - center
        diff_list.append(np.abs(diff))

    # threshold cho magnitude
    diff_stack = np.stack(diff_list, axis=0)
    T = diff_stack.mean()

    for p in range(P):
        theta = 2 * np.pi * p / P
        dy = int(round(R * np.sin(theta)))
        dx = int(round(R * np.cos(theta)))

        neighbor = img[pad+dy:pad+dy+h, pad+dx:pad+dx+w]
        diff = neighbor - center

        # CLBP_S
        CLBP_S |= ((diff >= 0).astype(np.uint8) << p)

        # CLBP_M
        CLBP_M |= ((np.abs(diff) >= T).astype(np.uint8) << p)

    # CLBP_C
    CLBP_C = (center >= center.mean()).astype(np.uint8)

    return CLBP_S, CLBP_M, CLBP_C

def hist(x, bins=256):
    h, _ = np.histogram(x.ravel(), bins=bins, range=(0, bins))
    return h

import numpy as np
import librosa
from skimage.feature import local_binary_pattern


def build_stft_dataset(
    wav_list,
    sr=16000,
    segment_sec=2,
    out_path="X_stft.npy"
):

    all_feats = []
    segment_len = int(segment_sec * sr)
    count = 0
    for wav_path in wav_list:

        print("Processing:", count + 1)
        count += 1

        y, sr = librosa.load(wav_path, sr=sr)

        # nếu audio < 2s
        if len(y) < segment_len:
            segments = [y]
        else:
            num_segments = len(y) // segment_len
            segments = [
                y[i*segment_len:(i+1)*segment_len]
                for i in range(num_segments)
            ]

        for segment in segments:

            stft_feat = extract_stft(segment, sr)

            CLBP_S, CLBP_M, CLBP_C = clbp(stft_feat, P=8, R=1)
            h_s = hist(CLBP_S)
            h_m = hist(CLBP_M)
            h_c = hist(CLBP_C, bins=2)

            feature = np.concatenate([h_s, h_m, h_c])
            all_feats.append(feature)

        print(f"{wav_path} -> {len(segments)} segments")

    X = np.vstack(all_feats)

    np.save(out_path, X)

    print("Saved:", out_path, X.shape)

    return X
import os
CLASS = "spoof"
DIR = r"E:\PythonFile\Project\Voice-Activity-Detect\data\processed\musan\speech"

wav_list = [
    os.path.join(DIR, f)
    for f in os.listdir(DIR)
    if f.endswith(".wav")
]
print(wav_list)
X_mfcc = build_stft_dataset(
    wav_list,
    out_path=f"E:/PythonFile/Project/Voice-Activity-Detect/data/feature/train/CLBP/stft_clbp_speech"
)

print(len(wav_list))
print(X_mfcc.shape)