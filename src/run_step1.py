from step1_light_xyz import extract_object_xyz
from step1_checks import run_all_checks

def main():
    # Extract light
    extract_object_xyz("dataset/bbox_light.csv", "outputs/light_xyz_cam.csv")
    run_all_checks("outputs/light_xyz_cam.csv", label="Traffic Light")

    # Extract golf cart
    extract_object_xyz("dataset/bbox_cart.csv", "outputs/cart_xyz_cam.csv")
    run_all_checks("outputs/cart_xyz_cam.csv", label="Golf Cart")


if __name__ == "__main__":
    main()
