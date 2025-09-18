import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def run_step3():
    # Load trajectory
    df = pd.read_csv("outputs/car_traj_world.csv")

    x = df["Xw"]
    y = df["Yw"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(x.min() - 2, x.max() + 2)
    ax.set_ylim(y.min() - 2, y.max() + 2)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Ego Car Trajectory in World Frame")
    ax.grid(True)

    # Plot elements
    line, = ax.plot([], [], lw=2, label="Car trajectory")
    point, = ax.plot([], [], "ro", label="Car position")
    ax.legend()

    # Update function
    def update(frame):
        x_data = x.iloc[:frame+1]
        y_data = y.iloc[:frame+1]
        line.set_data(x_data, y_data)
        if len(x_data) > 0:
            # Wrap single values in lists so Matplotlib sees them as sequences
            point.set_data([x_data.iloc[-1]], [y_data.iloc[-1]])
        return line, point

    # Animate
    fps = 15
    ani = animation.FuncAnimation(
        fig, update, frames=len(df), interval=1000/fps, blit=True
    )

    # Save as GIF with Pillow (no ffmpeg required)
    ani.save("outputs/trajectory.gif", writer="pillow", fps=fps)
    print("Animation saved to outputs/trajectory.gif")

if __name__ == "__main__":
    run_step3()
