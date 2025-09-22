# src/step1_checks.py
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_distance_time(df: pd.DataFrame, out="outputs/light_distance_vs_frame.png"):
    df = df.copy()
    df["dist_m"] = np.sqrt(df["X"]**2 + df["Y"]**2 + df["Z"]**2)
    plt.figure(figsize=(7,4))
    plt.plot(df["frame"], df["dist_m"], linewidth=2)
    plt.xlabel("frame")
    plt.ylabel("distance to light (m)")
    plt.title("Traffic light distance over time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out); plt.close()

def overlay_bboxes(bbox_csv, rgb_dir, sample_n=8, out="outputs/bbox_overlays.png"):
    b = pd.read_csv(bbox_csv).sort_values("frame")
    ids = np.linspace(0, len(b)-1, num=min(sample_n, len(b)), dtype=int)
    tiles = []
    for idx in ids:
        row = b.iloc[idx]
        fid = int(row["frame"])
        path = os.path.join(rgb_dir, f"left{fid:06d}.png")
        img = cv2.imread(path)
        if img is None: continue
        x1,y1,x2,y2 = map(int, [row.x1, row.y1, row.x2, row.y2])
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 3)
        u = int((x1 + x2)/2); v = int((y1 + y2)/2)
        cv2.circle(img, (u,v), 8, (0,0,255), -1)
        tiles.append(cv2.resize(img, (640, 400)))
    if tiles:
        # make a grid
        rows = []
        for i in range(0, len(tiles), 4):
            row = np.hstack(tiles[i:i+4])
            rows.append(row)
        grid = np.vstack(rows)
        cv2.imwrite(out, grid)

def scatter_xyz(df: pd.DataFrame, out="outputs/light_xyz_scatter.png"):
    plt.figure(figsize=(5,5))
    plt.scatter(df["X"], df["Y"], s=8)
    plt.xlabel("X (forward, m)")
    plt.ylabel("Y (right, m)")
    plt.title("Traffic light XY in camera frame")
    plt.axis("equal"); plt.grid(True)
    plt.tight_layout(); plt.savefig(out); plt.close()

def run_all_checks(csv_path, label="Object", bbox_csv=None, rgb_dir="dataset/rgb"):
    os.makedirs("outputs", exist_ok=True)
    df = pd.read_csv(csv_path)

    # 1) Basic stats
    ok_rate = (df["ok"]==1).mean()
    print(f"[{label}] Valid fraction: {ok_rate:.1%}")
    print(df[["X","Y","Z"]].describe())

    # 2) Distance vs frame
    plot_distance_time(df, out=f"outputs/{label.lower().replace(' ','_')}_dist_vs_frame.png")

    # 3) Overlay bbox centers (optional: only if bbox_csv provided)
    if bbox_csv is not None:
        overlay_bboxes(bbox_csv, rgb_dir, out=f"outputs/{label.lower().replace(' ','_')}_bbox_overlays.png")

    # 4) XY scatter in camera frame
    scatter_xyz(df, out=f"outputs/{label.lower().replace(' ','_')}_xyz_scatter.png")

    # 5) Print preview
    preview = df.head()
    preview = preview.assign(
        dist_m=np.sqrt(preview["X"]**2 + preview["Y"]**2 + preview["Z"]**2)
    )
    print(f"\nFirst 5 extracted {label} positions (camera frame):")
    print(preview[["frame","X","Y","Z","dist_m"]])

if __name__ == "__main__":
    run_all_checks()
