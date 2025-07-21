# Convert timestamps to milliseconds and gaze positions to degrees, remove
# samples during blink periods, and exclude recordings with too many NaNs or
# not enough samples

from argparse import ArgumentParser
from pathlib import Path
import re
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def normalize_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def make_output_dirs(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)


def get_subject_from_filename(file: Path) -> int:
    pattern = r"^S_(?P<subject_id>\d+)_.*$"
    match = re.match(pattern, file.stem)
    assert match is not None
    subject_id = match.groupdict()["subject_id"]
    return int(subject_id)


def load_recording(file: Path) -> pd.DataFrame:
    df = pd.read_csv(file)
    print("Loaded columns:", df.columns.tolist())
    print("First few rows:")
    print(df.head(3))  # Show first 3 rows
    numeric_cols = df.columns[df.columns != "StimulusTaskName"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    return df


def save_recording(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, na_rep="NaN", index=False)


def filter_stimulus(z: np.ndarray, task: pd.Series):
    invalid_position = (z == 0) | np.isnan(z)
    unwanted_task = task.isin(("BlinkStimulus", "None", "ReadingQuiz"))
    return ~invalid_position & ~unwanted_task


def ensure_minimum_validity(lx: np.ndarray, rx: np.ndarray, bx: np.ndarray, threshold: float = 0.5) -> bool:
    print("lx size:", lx.size, "rx size:", rx.size, "bx size:", bx.size)
    isnan = np.isnan(lx) | np.isnan(rx) | np.isnan(bx)
    print("isnan array sample:", isnan[:10])
    portion_nan = np.mean(isnan)
    print(f"→ Portion of invalid gaze rows: {portion_nan*100:.2f}%")
    return portion_nan <= threshold


def ensure_minimum_length(length: int, threshold: int = 2500) -> bool:
    return length >= threshold


def timestamp_to_relative_ms(t: np.ndarray) -> np.ndarray:
    t_ms = t * 1000
    t_ms_relative = t_ms - t_ms[0]
    return t_ms_relative


def l2norm(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    inverse_magnitude = 1.0 / np.sqrt(x**2 + y**2 + z**2)
    return x * inverse_magnitude, y * inverse_magnitude, z * inverse_magnitude


def atan2d(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.arctan2(y, x))


def vector_to_degrees(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    x, y, z = l2norm(x, y, z)
    x_deg = atan2d(x, np.sqrt(y**2 + z**2))
    y_deg = atan2d(y, z)
    return x_deg, y_deg


def convert_recording(df: pd.DataFrame, subject_id) -> Optional[pd.DataFrame]:

    print("TASK NAME:", df["StimulusTaskName"].iloc[0])

    task_name = df["StimulusTaskName"].iloc[0]

    #if task_name in ("Vergence", "SmoothPursuit", "RandomSaccades"):
     #   keep_index = filter_stimulus(
      #      df["StimulusPosition_Z"].to_numpy(), df["StimulusTaskName"]
       # )
        #df = df[keep_index]

    is_valid = ensure_minimum_validity(
        df["LeftGazeDirection_X"].to_numpy(),
        df["RightGazeDirection_X"].to_numpy(),
        df["CameraRaycast_X"].to_numpy(),
        threshold=0.5,
    )
    if not is_valid:
        print("Warning: Too many invalid samples")
        return None

    is_long_enough = ensure_minimum_length(len(df), threshold=2500)
    if not is_long_enough:
        print("Warning: Not enough samples")
        return None

    n = timestamp_to_relative_ms(df["Timestamp"].to_numpy())
    sid = np.full(len(df), subject_id)
    age = df["ParticipantAge"].to_numpy()
    imgid = df["ImageName"].to_numpy()
    hx = df["HeadPositionX"].to_numpy()
    hy = df["HeadPositionY"].to_numpy()
    hz = df["HeadPositionZ"].to_numpy()
    hrx = df["HeadRotationX"].to_numpy()
    hry = df["HeadRotationY"].to_numpy()
    hrz = df["HeadRotationZ"].to_numpy()

    lx, ly = vector_to_degrees(
        df["LeftGazeDirection_X"].to_numpy(),
        df["LeftGazeDirection_Y"].to_numpy(),
        df["LeftGazeDirection_Z"].to_numpy(),
    )
    rx, ry = vector_to_degrees(
        df["RightGazeDirection_X"].to_numpy(),
        df["RightGazeDirection_Y"].to_numpy(),
        df["RightGazeDirection_Z"].to_numpy(),
    )
    x, y = vector_to_degrees(
        df["CameraRaycast_X"].to_numpy(),
        df["CameraRaycast_Y"].to_numpy(),
        df["CameraRaycast_Z"].to_numpy(),
    )

    # Eye ball center in meters for each eye
    clx = df["LeftGazeBase_X"].to_numpy()
    cly = df["LeftGazeBase_Y"].to_numpy()
    clz = df["LeftGazeBase_Z"].to_numpy()
    crx = df["RightGazeBase_X"].to_numpy()
    cry = df["RightGazeBase_Y"].to_numpy()
    crz = df["RightGazeBase_Z"].to_numpy()

    has_target_xy = df["StimulusTaskName"].iloc[0] in ("Vergence", "SmoothPursuit", "RandomSaccades")
    # has_target_z = df["StimulusTaskName"].iloc[0] == "Vergence"
    xt, yt = vector_to_degrees(
        df["StimulusPosition_X"].to_numpy(),
        df["StimulusPosition_Y"].to_numpy(),
        df["StimulusPosition_Z"].to_numpy(),
    )
    zt = df["StimulusPosition_Z"].to_numpy()
    if not has_target_xy:
        xt[:] = np.nan
        yt[:] = np.nan
    # if not has_target_z:
    #     zt[:] = np.nan

    out_df = pd.DataFrame(
        {
            "n": n,
            "sid": sid,
            "age": age,
            "tn": task_name,
            "imgid": imgid,
            "hx": hx,
            "hy": hy,
            "hz": hz,
            "hrx": hrx,
            "hry": hry,
            "hrz": hrz,
            "x": x,
            "y": y,
            "lx": lx,
            "ly": ly,
            "rx": rx,
            "ry": ry,
            "xT": xt,
            "yT": yt,
            "zT": zt,
            "clx": clx,
            "cly": cly,
            "clz": clz,
            "crx": crx,
            "cry": cry,
            "crz": crz,
        }
    )
    return out_df


def round_decimals(df: pd.DataFrame, places: int = 4) -> pd.DataFrame:
    return df.round(decimals=places)


def convert_subject_recordings(files: Sequence[Path], dest: Path, subject_id) -> None:
    out_queue: List[Tuple[pd.DataFrame, Path]] = []
    for file in files:
        df = load_recording(file)
        out_df = convert_recording(df, subject_id)
        if out_df is None:
            print("Warning: Data requirements not met for", file.stem)
            return
        out_df = round_decimals(out_df, places=4)
        out_file = dest / file.name
        out_queue.append((out_df, out_file))
    
    for out_df, out_file in out_queue:
        save_recording(out_df, out_file)


def convert_all_recordings(src: Path, dest: Path) -> None:
    file_list = list(src.iterdir())
    subject_ids = [get_subject_from_filename(file) for file in file_list]
    unique_ids = np.unique(subject_ids)
    id_to_file_indices = {
        subject_id: np.where(subject_ids == subject_id)[0]
        for subject_id in unique_ids
    }

    for subject_id, file_indices in id_to_file_indices.items():
        subject_files = [file_list[i] for i in file_indices]
        print("----------")
        print("Converting files for subject", subject_id)
        convert_subject_recordings(subject_files, dest, subject_id)
        print("Finished converting subject", subject_id)


if __name__ == "__main__":
    parser = ArgumentParser(description="Clean up the raw GazeBaseVR data set.")
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="path to existing directory containing flattened, raw GazeBaseVR data",
    )
    parser.add_argument(
        "--dest",
        type=str,
        required=True,
        help="path to output directory for storing converted data",
    )
    args = parser.parse_args()

    input_dir = normalize_path(args.src)
    output_dir = normalize_path(args.dest)
    
    make_output_dirs(output_dir)
    convert_all_recordings(input_dir, output_dir)
