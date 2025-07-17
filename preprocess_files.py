import pandas as pd
import os
import collections
from argparse import ArgumentParser
from pathlib import Path
from typing import Union

# Returns a clean, absolute file path
def normalize_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()

# Creates directory if directory doesn't exist
def make_output_dirs(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)

# Pads each ID in series with leading 0's to ensure minimum length of 3 characters
def ensureIDDigits(pid):
    return pid.astype(str).str.zfill(3)

# Splits file name and returns participant ID and round number
def parse_filename(file_name):
    try:
        parts = file_name.split('_')
        rxxx = parts[1][1:]  # e.g., 3002

        return rxxx
    except:
        return None

# Creates list of all files from round 1
def createListOfR1Files(folder):
    list_of_files = []
    for file in folder:
        if "S_1" in file:
            list_of_files.append(file)
    return list_of_files

# Starts processing for all files from round 1
def ProcessAndSaveAllR1Files(norm_dir, round_one_dir, round_one_files):
    valid_dfs, valid_files = getValidR1Files(round_one_dir, round_one_files)
    normalizeAndSave(norm_dir, valid_dfs, valid_files)

    return valid_files

# Creates list of valid df's and files after removing NaN rows (valid if > 1250 or has more than 90% of its original length)
def getValidR1Files(round_1_dir, round_one_files):
    valid_dfs = []
    valid_files = []

    for file in round_one_files:
        print("Processing: ", file)
        df_raw = pd.read_csv(os.path.join(round_1_dir, file))
        df_raw = df_raw[['n', 'clx', 'cly', 'clz', 'crx', 'cry', 'crz']]

        original_len = len(df_raw)
        df = df_raw.dropna()

        # Check if csv file has 1250 rows and has more than 90% valid rows
        if len(df) < 1250 or len(df) < 0.9 * original_len:
            print(f"Skipped (bad quality): {file}")
            continue

        print("Finished selecting valid rd 1 file: " + file)
        valid_dfs.append(df)
        valid_files.append(file)
    
    return valid_dfs, valid_files

# Normalizes all values using min and max across entire dataset
def normalizeAndSave(norm_dir, valid_dfs, valid_files):
    min_vals, max_vals = getMinMax(valid_dfs)

    for df, file in zip(valid_dfs, valid_files):
        df['n'] = df['n'] / 1000.0
        for col in ['clx', 'cly', 'clz', 'crx', 'cry', 'crz']:
            df[col] = 2 * ((df[col] - min_vals[col]) / (max_vals[col] - min_vals[col])) - 1
        df.to_csv(os.path.join(norm_dir, file), index=False)
        print("Saved:", file)

# Copies all round 1 files to separate folder
def moveAllR1Files(input_dir, round_1_dir):
    gaze_files = os.listdir(input_dir)
    for file in createListOfR1Files(gaze_files):
        print("Moving: " + file)
        df = pd.read_csv(os.path.join(input_dir, file)) # 1.Data shape: [num_rows, 7]
        df = df[['n', 'clx', 'cly', 'clz', 'crx', 'cry', 'crz']]
        df.to_csv(os.path.join(round_1_dir, file), index=False)
        print("Successfully moved: " + file)

# Gets min and max value across entire dataset
def getMinMax(valid_dfs):
    big_df = pd.concat(valid_dfs)
    return big_df.min(), big_df.max()

# Removes leading leading digits - Only returns id (ex - 3021 -> 21)
def removeLeadingNumbers(user_id):
    return user_id[1:].lstrip('0')

# Splits each valid file into non-overlapping windows of 1250 rows - Skips any file with fewer than 1250 rows.
def windowData(norm_dir, valid_files):
    print("Windowing Data...")
    x = []
    y = []
    for file in valid_files:
        file_path = os.path.join(norm_dir, file)
        df = pd.read_csv(file_path)

        if len(df) < 1250:
            continue

        user_id = parse_filename(file)

        i = 0
        for i in range(0, len(df) - 1250 + 1, 1250):
            window = df.iloc[i:i+1250].values
            x.append(window)
            y.append(user_id)
    print("Finished windowing data")
    return x, y

# Returns inputs and labels
def getXY(input_dir, round_1_dir, norm_dir):
    if len(os.listdir(round_1_dir)) == 0:
        moveAllR1Files(input_dir, round_1_dir)
    if len(os.listdir(norm_dir)) == 0:
        round_1_files = os.listdir(round_1_dir)
        valid_files = ProcessAndSaveAllR1Files(norm_dir, round_1_dir,createListOfR1Files(round_1_files))
    else:
        valid_files = [f for f in os.listdir(norm_dir) if f.endswith('.csv')]
    x, y = windowData(norm_dir, valid_files)
    label_to_index = {label: idx for idx, label in enumerate(sorted(set(y)))}
    y = [label_to_index[label] for label in y]

    label_counts = collections.Counter(y)

    unique_labels = set(y)

    return x, y

def printLabelInfo(y):
    print("Sample of labels:", y[:10])
    print("Label type:", type(y[0]))
    print("Unique labels count:", len(set(y)))
    print("Min label:", min(map(int, y)))
    print("Max label:", max(map(int, y)))

if __name__ == "__main__":
    parser = ArgumentParser(description = "Preprocess round 1 files from GazeBaseVR data set")
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Path to existing directory containing GazeBaseVR data",
    )
    parser.add_argument(
        "--round_1_dir",
        type=str,
        required=True,
        help="Path to output directory for storing round 1 files"
    )
    parser.add_argument(
        "--norm_dir",
        type=str,
        required=True,
        help="path to output directory for storing normalized data"
    )
    args = parser.parse_args()

    input_dir = normalize_path(args.src)
    round_1_dir = normalize_path(args.round_1_dir)
    norm_dir = normalize_path(args.norm_dir)

    make_output_dirs(round_1_dir)
    make_output_dirs(norm_dir)
    x, y = getXY(input_dir, round_1_dir, norm_dir) # 2. X-Shape: [1250, 7] Y-Shape: [labels]
    printLabelInfo(y)

