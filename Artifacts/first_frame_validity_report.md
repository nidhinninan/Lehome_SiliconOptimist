### **Technical Report: First Frame Validity & Sanity Check Methodology**

---

#### **1. Analysis of the "First Frame" Guarantee**
In the **LeHome/LeRobot** ecosystem, there is **no absolute hardware/software guarantee** that Frame 0 captures the entire cloth in a perfectly flattened state. However, the following factors make it the most reliable candidate:

*   **Teleoperation Convention:** Expert demonstrations (like those in `lehome/dataset_challenge_merged`) almost always begin with a "reset" phase where the garment is spawned/placed on the table and the robot is at a neutral home position.
*   **Dataset Metadata:** The `garment_info.json` file (referenced in `dataset_inspection.py`) specifically logs `object_initial_pose` at the start of episodes. This confirms that the simulation records a distinct "Initial State" before manipulation begins.
*   **Evaluation Protocol:** The official evaluation script loads the garment and *then* starts the policy. This implies the first observation the policy receives is the static, unfolded state.

**Risk Factor:** If the expert began the recording *after* the first grasp, Frame 0 will show a partially occluded cloth. This is why a sanity check is mandatory.

---

#### **2. Proposed Sanity Check: The "Random $k$-th Frame" Sampler**
To verify that the classifier will be trained on informative images, you can implement a script to extract a random cross-section of frames.

**The Strategy:**
1.  **Library:** Use the `lerobot` Python API (specifically `LeRobotDataset`). This is significantly faster than manually parsing Parquet/MP4 files because it handles the video decoding and synchronization internally.
2.  **Logic:**
    *   Load a specific dataset (e.g., `top_long_merged`).
    *   Randomly sample $n$ episode indices (e.g., 20 random episodes).
    *   Fetch the $k$-th frame (where $k=0$ for the start) from all three cameras (`top`, `left`, `right`).
    *   Save these as a grid or a folder of JPEGs for human review.

**Sample Implementation Logic (Python):**
```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import torch
from torchvision.utils import save_image

# 1. Load the dataset
dataset = LeRobotDataset("path/to/top_long_merged")

# 2. Pick N random episodes
num_samples = 10
k_frame = 0 # The frame we want to check
indices = torch.randint(0, dataset.num_episodes, (num_samples,))

# 3. Extract and save
for i, ep_idx in enumerate(indices):
    # Fetch the specific frame for that episode
    # Note: dataset[i] usually returns a dict of tensors
    frame_data = dataset.get_item(ep_idx * dataset.fps) # Approximation for k-th frame
    
    # Save the 'top_rgb' view for inspection
    img = frame_data["observation.images.top_rgb"]
    save_image(img, f"sanity_check_ep{ep_idx}_frame{k_frame}.jpg")
```

---

#### **3. Recommendation for the Visual Classifier**
If the sanity check reveals that Frame 0 is inconsistent (e.g., the robot arm is blocking the view in 10% of cases), you should:
1.  **Temporal Pooling:** Instead of just Frame 0, train the classifier on a "Max-Pool" or average of Frames 0–5. This ensures that any momentary occlusion doesn't break the routing logic.
2.  **Camera Selection:** Rely primarily on the **`top_rgb`** camera for the classifier. The `left` and `right` wrist cameras are often too close to the cloth at t=0 to capture the full geometry needed for category identification.
