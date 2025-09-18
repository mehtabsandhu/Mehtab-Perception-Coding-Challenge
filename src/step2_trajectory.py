import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def compute_rotation(light0):
    Z = np.array([0, 0, 1])
    dir0 = np.array([light0[0], light0[1], 0.0])
    norm = np.linalg.norm(dir0)
    if norm < 1e-6:
        raise ValueError("Initial light vector is degenerate (too close to Z-axis)")
    Xw = dir0 / norm
    Yw = np.cross(Z, Xw)
    R = np.vstack([Xw, Yw, Z]).T
    return R

def run_step2():
    df = pd.read_csv("outputs/light_xyz_cam.csv")

    # Interpolate missing values
    df[["X", "Y", "Z"]] = df[["X", "Y", "Z"]].interpolate()
    df[["X", "Y", "Z"]] = df[["X", "Y", "Z"]].fillna(method="bfill").fillna(method="ffill")

    light0 = df[["X", "Y", "Z"]].iloc[0].to_numpy()
    R = compute_rotation(light0)

    car_positions = []
    for frame, row in df.iterrows():
        light_cam = row[["X", "Y", "Z"]].to_numpy()
        car_cam = -light_cam
        car_world = R @ car_cam
        car_positions.append([frame, *car_world])

    car_positions = np.vstack(car_positions)
    out_df = pd.DataFrame(car_positions, columns=["frame", "Xw", "Yw", "Zw"])
    os.makedirs("outputs", exist_ok=True)
    out_df.to_csv("outputs/car_traj_world.csv", index=False)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.plot(out_df["Xw"], out_df["Yw"], marker="o", markersize=2, label="Car trajectory")
    plt.scatter([0], [0], c="red", marker="*", s=100, label="Traffic light")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True)
    plt.title("Ego Car Trajectory in World Frame")
    plt.legend()
    plt.savefig("outputs/trajectory.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    run_step2()