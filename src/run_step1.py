# src/run_step1.py
from step1_light_xyz import extract_all_xyz
from step1_checks import run_all_checks

def main():
    extract_all_xyz()       # writes outputs/light_xyz_cam.csv
    run_all_checks()        # writes quick plots into outputs/

if __name__ == "__main__":
    main()
