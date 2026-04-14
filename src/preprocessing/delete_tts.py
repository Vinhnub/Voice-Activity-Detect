import os
import shutil

BASE_PATH = r"E:\Pythonfile\Voice-Activity-Detect\data\raw\LA"

ALLOWED_ATTACKS = {
    "A01", "A02", "A03", "A04",
    "A05", "A06", "A07", "A08",
    "A09", "A10", "A11", "A12",
    "A13", "A14", "A15", "A16",
    "A17", "A18", "A19"
}

SETS = {
    "train": {
        "protocol": os.path.join(BASE_PATH,
            "ASVspoof2019_LA_cm_protocols",
            "ASVspoof2019.LA.cm.train.trn.txt"),
        "flac": os.path.join(BASE_PATH,
            "ASVspoof2019_LA_train",
            "flac")
    },
    "dev": {
        "protocol": os.path.join(BASE_PATH,
            "ASVspoof2019_LA_cm_protocols",
            "ASVspoof2019.LA.cm.dev.trl.txt"),
        "flac": os.path.join(BASE_PATH,
            "ASVspoof2019_LA_dev",
            "flac")
    },
    "eval": {
        "protocol": os.path.join(BASE_PATH,
            "ASVspoof2019_LA_cm_protocols",
            "ASVspoof2019.LA.cm.eval.trl.txt"),
        "flac": os.path.join(BASE_PATH,
            "ASVspoof2019_LA_eval",
            "flac")
    }
}


def filter_and_split(protocol_path, flac_folder):

    bonafide_folder = os.path.join(flac_folder, "bonafide")
    spoof_folder = os.path.join(flac_folder, "spoof")

    os.makedirs(bonafide_folder, exist_ok=True)
    os.makedirs(spoof_folder, exist_ok=True)

    bonafide_count = 0
    spoof_count = 0
    removed_count = 0

    # Đọc protocol và tạo mapping
    keep_map = {}

    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            utt_id = parts[1]
            label = parts[-1]

            attack = None
            if label == "spoof":
                attack = parts[-2]

            filename = utt_id + ".flac"

            if label == "bonafide":
                keep_map[filename] = "bonafide"

            elif attack in ALLOWED_ATTACKS:
                keep_map[filename] = "spoof"

    # Duyệt file thực tế
    for filename in os.listdir(flac_folder):

        if not filename.endswith(".flac"):
            continue

        src = os.path.join(flac_folder, filename)

        if filename in keep_map:
            print(filename)

            if keep_map[filename] == "bonafide":
                dst = os.path.join(bonafide_folder, filename)
                bonafide_count += 1
            else:
                dst = os.path.join(spoof_folder, filename)
                spoof_count += 1

            shutil.move(src, dst)

        else:
            os.remove(src)
            removed_count += 1

    print(f"\nProcessed: {flac_folder}")
    print(f"Bonafide: {bonafide_count}")
    print(f"Spoof: {spoof_count}")
    print(f"Removed: {removed_count}")


# ======================
# RUN
# ======================

for name, paths in SETS.items():
    filter_and_split(paths["protocol"], paths["flac"])

print("\nDataset filtered and split successfully!")