# assigned user permission correctly

- probably don't need below cmd any more as I have updated the .sh file (had the use principia instead of $USER in the .sh script to correctly give permissions)
```python
sudo chown -R principia:principia /data/lehome_workspace
```
- use rsync with the '--exclude tags' because we don't want to loss the auto generated files because of uv sync and other cmds from the install .sh file.

rd# DINOv2 Adpaption
- DINOv2 patch tokens preserve fine-grained spatial info - even paper say reach SOTA in AED20K with on patchh token. CLS is helpful for global patch tokens (dense tasks like semantic segmentation).
- current path has has no spatial readout ; u-net only see flat global vectors.
- loss all patch info as they are discards and only CLS is kept.
## appraoches in the literature
- Voltron (paper) has a MAP head atht have shown double the success rate than a direct single token extraction.
- or use a FiLM conditioned U-Net (DINOv3 + Diffusion Policy (arXiv 2509.17684) )
- keeping dense map better as SpatialSoftmax will collapse the 2k scalars and paper("Spotlighting Task-Relevant Features" (arXiv 2601.21416) ) shows
## inti proposal risks
- can introduce spurious peaks as DINOv2 creates scratchpad patch tokens in low information spaces - uses Spatial Softmax (due to 2k reductions) will lock onto these peaks and create reproducible failure mode
- orig SpatialSoftmax relies only works with last layer (local receptive field) but ViT has multiple layers and only taking the last layer to SpatialSoftmax will not encapsulate all the spatial info.
- course grid (16*16) will not capture the fine-pixel level creases and wrinkles needed to understand feature to assist with learning policy.
- 98K dims down to 2·K : significant coord squeeze

- video backend with "pyav" can be slow so install FFmpeg
- current setup not necessary executable on any VM system
- save file to gdrive to save mempory (make sure not to select auto-config on VM system)
- sneakly bash script error: In bash, if you place a comment (#) inside a command that uses backslashes (\) for line continuation, it breaks the continuation.
 
- MAP head is a high level count of ho many "features' the models is paying attention to. OFr

- had oom errors : reduce the eval state and made it very light only want a vague understanding of training progress.

## personal mistake / observation
- **mistake**: assumed “slow training” meant GPU saturation; logs showed `data_s` > `updt_s` → loader/IO/decode bound, not compute-bound.
- **target**: tune workers/pipeline/storage before chasing bigger batches or model tricks for wall time.
- **mistake**: treating +4 workers (20→24) as a big win — realistic savings often single-digit % unless `data_s` drops clearly in logs.
- **target**: confirm imbalance with `nvidia-smi --query-gpu=utilization.gpu -l 1` (sample → avg); low avg util + high `data_s` = feed the GPU first.
- **mistake**: `checkpoints/last` as a real directory (not a symlink) → `FileExistsError` when LeRobot tries `symlink_to` for a new step; training dies *after* a save attempt.
- **target**: `last` must be a symlink; if it became a directory, rename to `050000` (etc.) and `ln -sfn <step> last`.
- **mistake**: resume smoke test without `--output_dir=...` in `EXTRA_TRAIN_ARGS` — LeRobot kept writing checkpoints to the path baked into `train_config.json`, not the `OUTPUT` you passed to the shell script.
- **target**: to isolate smoke, set `--output_dir` to the smoke tree explicitly; verify log line `Output dir:` matches intent.
- **mistake**: `ENABLE_RCLONE_CHECKPOINT_SYNC=1` + `rclone move` + `--min-age` — old-enough step dirs (incl. the one `last` points at) got moved off disk before train opened `train_config.json` → `FileNotFoundError`; looked like “rm smoke broke training” but was rclone.
- **target**: disable rclone on resume until active checkpoint is local, or exclude the `last` target from move (script patch); double-check Drive path: `LeHome/models/<project>/<JOB_NAME>/<RUN_TAG>/checkpoints/` — `JOB_NAME` and `RUN_TAG` must match the run that uploaded, not a guessed folder.
- **mistake**: grepping “core” / `find … name core*` under venv → thousands of `core.py` hits; not crash dumps.
- **target**: Ubuntu uses Apport (`core_pattern` pipe); check `/var/crash`; narrow finds to `^core$` / `vgcore.*` outside site-packages.
- **observation**: `pretrained_path: null` in resumed `train_config.json` can still be fine — weights load from the checkpoint tree + `training_state/`; trust presence of `training_step.json` and `last` resolution.