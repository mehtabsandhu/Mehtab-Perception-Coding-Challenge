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

def transform_to_world(csv_in, csv_out, R):
    df = pd.read_csv(csv_in)
    df[["X","Y","Z"]] = df[["X","Y","Z"]].interpolate().bfill().ffill()

    records = []
    for _, row in df.iterrows():
        P_cam = row[["X","Y","Z"]].to_numpy()
        P_world = R @ P_cam
        records.append([row["frame"], *P_world])

    out_df = pd.DataFrame(records, columns=["frame","Xw","Yw","Zw"])
    os.makedirs("outputs", exist_ok=True)
    out_df.to_csv(csv_out, index=False)
    return out_df

def run_step2():
    # --- Rotation matrix from the traffic light ---
    light_df = pd.read_csv("outputs/light_xyz_cam.csv")
    light_df[["X", "Y", "Z"]] = light_df[["X","Y","Z"]].interpolate().bfill().ffill()
    light0 = light_df[["X", "Y", "Z"]].iloc[0].to_numpy()
    R = compute_rotation(light0)

    # --- Ego car (negated light vector) ---
    car_df = transform_to_world(
        "outputs/light_xyz_cam.csv", "outputs/car_traj_world.csv", R
    )

    # --- Golf cart (direct transform) ---
    cart_df = transform_to_world(
        "outputs/cart_xyz_cam.csv", "outputs/cart_traj_world.csv", R
    )

        # --- Ego-only plot (required output) ---
    plt.figure(figsize=(6, 6))
    plt.plot(car_df["Xw"], car_df["Yw"], marker="o", markersize=2, color="blue", label="Ego Car")
    plt.scatter([0], [0], c="red", marker="*", s=120, label="Traffic Light")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True)
    plt.title("Ego Car Trajectory in World Frame (BEV)")
    plt.legend()
    plt.savefig("outputs/trajectory.png", dpi=150)   # <--- required file
    plt.close()

if __name__ == "__main__":
    run_step2()