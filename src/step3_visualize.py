import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def run_step3():
    # --- Load trajectories ---
    car_df = pd.read_csv("outputs/car_traj_world.csv")
    cart_df = pd.read_csv("outputs/cart_traj_world.csv")

    # Determine plotting limits (expand to include both objects)
    x_all = pd.concat([car_df["Xw"], cart_df["Xw"], pd.Series([0])])
    y_all = pd.concat([car_df["Yw"], cart_df["Yw"], pd.Series([0])])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(x_all.min() - 2, x_all.max() + 2)
    ax.set_ylim(y_all.min() - 2, y_all.max() + 2)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Ego Car & Golf Cart Trajectories (World Frame, BEV)")
    ax.grid(True)

    # Plot elements for ego car
    line_car, = ax.plot([], [], lw=2, color="blue", label="Ego Car")
    point_car, = ax.plot([], [], "bo")

    # Plot elements for golf cart
    line_cart, = ax.plot([], [], lw=2, color="green", label="Golf Cart")
    point_cart, = ax.plot([], [], "go")

    # Traffic light (fixed at origin)
    ax.scatter([0], [0], c="red", marker="*", s=120, label="Traffic light")

    ax.legend()

    # --- Update function for animation ---
    def update(frame):
        # Ego car
        x_car = car_df["Xw"].iloc[:frame+1]
        y_car = car_df["Yw"].iloc[:frame+1]
        line_car.set_data(x_car, y_car)
        if len(x_car) > 0:
            point_car.set_data([x_car.iloc[-1]], [y_car.iloc[-1]])

        # Golf cart
        x_cart = cart_df["Xw"].iloc[:frame+1]
        y_cart = cart_df["Yw"].iloc[:frame+1]
        line_cart.set_data(x_cart, y_cart)
        if len(x_cart) > 0:
            point_cart.set_data([x_cart.iloc[-1]], [y_cart.iloc[-1]])

        return line_car, point_car, line_cart, point_cart

    # --- Animate ---
    fps = 15
    ani = animation.FuncAnimation(
        fig, update, frames=max(len(car_df), len(cart_df)), interval=1000/fps, blit=True
    )

    ani.save("submission/trajectory_multi.gif", writer="pillow", fps=fps)
    print("Animation saved to outputs/trajectory_multi.gif")


if __name__ == "__main__":
    run_step3()
