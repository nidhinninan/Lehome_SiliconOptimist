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