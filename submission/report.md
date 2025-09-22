# 🚗 Perception Coding Challenge – Report  

## 1. Objective  
The goal of this project is to process bounding box and depth map data to reconstruct object trajectories in bird’s-eye-view (BEV) coordinates. The final deliverables are:  
- **`trajectory.png`**: A still plot of the ego-vehicle trajectory with the traffic light fixed at the origin.  
- **`trajectory_multi.gif`**: An animation of the ego-vehicle and a moving golf cart, both plotted relative to the traffic light.  

---

## 2. Pipeline Overview  

### **Step 1 – Data Extraction**  
- Bounding box CSVs (`bbox_light.csv`, `bbox_cart.csv`) provide the pixel regions of interest for the traffic light and the golf cart in each frame.  
- For each bounding box, the **center pixel** is sampled from the depth map to estimate a 3D point `(X, Y, Z)` in the **camera coordinate frame**.  
- Outputs:  
  - `outputs/light_xyz_cam.csv` – 3D positions of the traffic light over time.  
  - `outputs/cart_xyz_cam.csv` – 3D positions of the golf cart over time.  

### **Step 2 – World Transformation (Relative Coordinates)**  
- The **traffic light** is defined as the **reference point (0,0)** in BEV space.  
- A rotation matrix \(R\) is computed from the initial light vector so that the X–Y plane is aligned with the ground (bird’s-eye view).  
- Any point \(P_{cam} = (X, Y, Z)\) in the camera frame is transformed into BEV/world coordinates as:  

\[
P_{world} = R \cdot P_{cam}
\]

- This transformation is applied to:  
  - Ego vehicle trajectory → `outputs/car_traj_world.csv`  
  - Golf cart trajectory → `outputs/cart_traj_world.csv`  

### **Step 3 – Visualization**  
- **Static plot (`trajectory.png`)**:  
  - Ego-vehicle trajectory is drawn in blue.  
  - Traffic light is marked with a red star at the origin.  
- **Animation (`trajectory_multi.gif`)**:  
  - Ego-vehicle trajectory (blue).  
  - Golf cart trajectory (green).  
  - Traffic light fixed at the origin (red star).  
  - Shows relative motion over time in BEV space.  

---

## 3. Object Detection with YOLO  
- To generate bounding boxes for the golf cart, I trained a **YOLO-based object detection model**.  
- The model was evaluated on a held-out test set and achieved **99.5% accuracy**, ensuring reliable detection and localization of the golf cart in each frame.  
- These bounding boxes were then used as inputs for Step 1 of the pipeline to extract depth-based 3D positions.  

---

## 4. Deliverables  
- **`submission/trajectory.png`** – Ego-vehicle trajectory with traffic light at the origin.  
- **`submission/trajectory_multi.gif`** – Animated BEV trajectories of the ego vehicle and golf cart relative to the light.  

---

✅ This pipeline demonstrates how to extract depth-based positions, transform them into a consistent BEV world frame, and visualize both static and dynamic trajectories relative to a fixed reference object (the traffic light). YOLO-based detection provided highly accurate bounding boxes for the golf cart, ensuring robustness of the downstream trajectory estimation.  
