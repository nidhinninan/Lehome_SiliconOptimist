<!-- IGNORE: DO NOT EDIT OR ADD TO THIS FILE MANUALLY -- UNLESS I EXPLICITLY ASK YOU TO ADD TO THIS SPECIFIC FILE BY NAME (e.g., "add to Learning.md-->

- For DP ususally train each dataset separately
- to check for maximisiing the batch size run "lerobot-train --config_path=configs/train_dp.yaml \
  --batch_size=24 \
  --steps=50" -> this is to see if we hit OOM.

- When a training run failed or Ctrl+C, find the exact PID as the memory gets locked. Find Pid and kill it 
```bash
nvidia-smi
# find PID under processes
kill -9 <PID>
```

- Even though AMP makes the GPU calculate faster, batch size 12 introduces so many images that your CPU dataloader completely chokes trying to decode them all. Processing 12 images takes more than twice as long as processing 8 images. (No AMP, BS=8) processed steps at 0.53 seconds and (AMP, BS=12) processed steps at 1.16 seconds

## 🚨 Final Submission Gotcha: Randomized Loading
*   **The Problem:** The official evaluation script loads garments from **different categories randomly** and does **NOT** give the policy a label (like "top_long").
*   **The Conflict:** If you train 4 separate DP models, you won't know which one to "turn on" in the middle of a random test run.
*   **The Solutions:**
    1.  **Unified Training:** Train one DP model on all 4 datasets combined (harder to converge).
    2.  **Visual Classifier:** Train a tiny classifier to look at the first frame, guess the garment type, and then load the correct DP weights.
    3.  **VLA/Foundation Model:** Use X-VLA or SmolVLA, which inherently handle task descriptions and visual variety.

- In pytorch, You can change the num_workers and essentially the num-workers is the number of unit processes that would be assigned to each core concurrently to serve data to GPU.  
- Is it has to be the maximum of the number of cores that your system has which means if it has 8 cores it can't go any more than 8. If it has 4 cores it can't go any more than 4. (IT CAN also be 3-4x the batch size but never than the number of cores)
- on the RTX a4000 setup I am using, I AM NOT limited by the GPU size - am limited by the 4 vcpu because the CPU always takes longer (3 x 490*640 image streams) to deliver the data sets to the GPU no matter if you run it on mixed precision "amp" as true or false.

## First long DP run
```
mkdir -p /data/lehome_workspace/lehome-challenge/logs/readout && \
/data/lehome_workspace/lehome-challenge/.venv/bin/python \
  -m lerobot.scripts.lerobot_train \
  --config_path=/data/lehome_workspace/lehome-challenge/configs/train_dp.yaml \
  --dataset.root=/data/lehome_workspace/lehome-challenge/Datasets/example/top_short_merged \
  --output_dir=/data/lehome_workspace/lehome-challenge/outputs/train/dp_top_short \
  --num_workers=4 \
  --policy.use_amp=false \
  --batch_size=8 \
  --steps=150000 2>&1 | tee /data/lehome_workspace/lehome-challenge/logs/readout/dp_top_short_training.log

```

## gripper + cloth contact inprovement
- Gripper-cloth slippage is a very common issue; most comon solution are changes at the silulator end, like changing the max_friction_combo between the gripper finders and cloth, collider type, and improving the solver stablity by increasing the max_position_iteration_count and physics frequency.
- API for PhysX - bind the cloth and gripper on contact programmatically
- Nodal Control - override the position/velocity of the cloth particles inside the gripper volume to match the gripper's transform every step

## In LeRobot DP training LR starving off
In the initial run , we had set the step size to about 30K - this meant that learning rate went from highest to 0 in that 30k run. Any longer training that that (on a policy meant to wrap up training at 30K) means that the learning rate is 0 for the rest of the training. -> no learning will happen even if you extend the training using --resume_from_checkpoint
- So second time (top_short), run it for 150k and see if convergence happens at around 100K and if the ressults look like it has setted off than wrap up there or else just resume till 150K.
- Appartently, Loss below ~0.020 has historically correlated with usable policies for cloth manipulation tasks (not sure why)
-- norms > 10–100 signal trouble -> model in chaotic region: loss stikes and training divergese
-- Norms < 0.01 mean the optimizer has no direction to go.
- the config i use; also has an explicit "grad_clip_norm: 10.0" to prevent exploding gradients 

-DP predict 16 future steps but execute 8 and then starts planning for next 16.

- the original eval scripts (util and evaluation.py) are bugged - they overwrite videos because the episode index resets to 0. so added a patch to save for "1 set of videos" for each garment type.

- run eval in parallel for individual cloths but adding a new custom_list_path for each garment type -> then i generated a new .sh file to split the exist cloth specific .txt under Release in Assets that showed make garments are there in each type. The .sh will split the .txt into 4 chunks and run the eval in parallel for each chunk. (see split_and_run_parallel.sh)

# Ways to impove DP training
- Use a pre-trained vision backbone. Help training start with some understanding of edges, corners, and basic shapes. -
- Add data augmentation (LeRobot allow for CLI override so we can add it as we resume training from a checkpoint that we interrupted before reaching full step we started the training to complete) -> just using the --dataset.image_transforms.enable=true with enable 6 different augmentation (w/ i think 3 at a time max); check the augent artifact for exact numbers
- New LeRobot 0.4.x+ uses draccus instead of hydra for config management. (what that means is that we need explicit -- fot config (look in to other diffs as well))
- Even with context switching, using CPU works is better - workload (image transforms) is so CPU-intensive that having more workers "in the queue" to feed the GPU outweighs the overhead of the CPU being over-subscribed. 

## log saving
- use " 2>&1 | tee /data/lehome_workspace/lehome-challenge/logs/<name>.log" to save the log. here 2&>1 means redirect both staderr and std out to the file and tee especially split to so that the readout is saved in file and shown in terminal.