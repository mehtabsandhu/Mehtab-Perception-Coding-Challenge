# src/step1_light_xyz.py
import os
import numpy as np
import pandas as pd

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
RGB_DIR = os.path.join(DATASET_DIR, "rgb")
XYZ_DIR = os.path.join(DATASET_DIR, "xyz")
BBOX_CSV = os.path.join(DATASET_DIR, "bbox_light.csv")

def load_xyz(frame: int) -> np.ndarray:
    path = os.path.join(XYZ_DIR, f"depth{frame:06d}.npz")
    arr = np.load(path)["xyz"][..., :3]
    return arr

def bbox_center(row) -> tuple[int, int]:
    u = int((row["x1"] + row["x2"]) / 2)
    v = int((row["y1"] + row["y2"]) / 2)
    return u, v

def valid_mask(P: np.ndarray) -> np.ndarray:
    # P: (N,3)
    if P.size == 0:
        return np.zeros((0,), dtype=bool)
    ok = np.isfinite(P).all(axis=1)
    ok &= ~(np.all(P == 0, axis=1))
    return ok

def robust_xyz_at(points_hw3: np.ndarray, u: int, v: int, patch_sizes=(3,5,9,13)) -> np.ndarray | None:
    """
    Return robust (X,Y,Z) around (u,v) by median of a patch.
    Expands patch until it finds enough valid points; returns None if all fail.
    """
    H, W, _ = points_hw3.shape
    for k in patch_sizes:
        r = k // 2
        y0, y1 = max(0, v - r), min(H, v + r + 1)
        x0, x1 = max(0, u - r), min(W, u + r + 1)
        patch = points_hw3[y0:y1, x0:x1].reshape(-1, 3)
        m = valid_mask(patch)
        vals = patch[m]
        if vals.shape[0] >= max(5, k):  # need a few valid samples
            # Remove gross outliers via IQR on depth (X forward)
            X = vals[:, 0]
            q1, q3 = np.percentile(X, [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            keep = (X >= lo) & (X <= hi)
            vals = vals[keep] if keep.any() else vals
            return np.median(vals, axis=0)
    return None

def extract_object_xyz(bbox_csv, save_csv, max_frames=None) -> pd.DataFrame:
    os.makedirs("outputs", exist_ok=True)
    bboxes = pd.read_csv(bbox_csv)

    if max_frames is not None:
        bboxes = bboxes.sort_values("frame").head(max_frames)

    records = []
    for _, row in bboxes.iterrows():
        fid = int(row["frame"])
        u = int((row["x1"] + row["x2"]) / 2)
        v = int((row["y1"] + row["y2"]) / 2)
        xyz = robust_xyz_at(load_xyz(fid), u, v)
        if xyz is None:
            rec = dict(frame=fid, X=np.nan, Y=np.nan, Z=np.nan, ok=0)
        else:
            rec = dict(frame=fid, X=float(xyz[0]), Y=float(xyz[1]), Z=float(xyz[2]), ok=1)
        records.append(rec)

    df = pd.DataFrame.from_records(records).sort_values("frame")
    df.to_csv(save_csv, index=False)
    return df


if __name__ == "__main__":
    extract_object_xyz("dataset/bbox_light.csv", "outputs/light_xyz_cam.csv")
    extract_object_xyz("dataset/bbox_cart.csv", "outputs/cart_xyz_cam.csv")
