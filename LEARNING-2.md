# Classifier
- For lazy loading the classifier, we only need to load it once every episode so the policy switching is not too big a concern as far as time loss goes.
- Other approach; load all 4 on the VRAM so that can don't have the load from "memory to VRAM" overhead

# KArpathy's autoResearch
- time contsrainted wall clock anaylsis - core design ; wall clock based parameter sweep for the most efficient training apprach (reduce having to do a full training to compare models), use a new vocbulary independant metric known as "Bits per Byte" to caompare models of different "architectures"
- ONLY EDIT The train.py file so that the infratrcuture is immutable.
- Cant do a simple hack together to utilize the insight and setup from AutoResearch in the LeRobot based Lehome repo setup AS lerobot is "step based" - SO wrapper approach without changing the underlying train.py equivalent will not work.
-- The lerobot only does evaluation after set number of steps but the Karpathy test apprach needs to do a "time based' loop test condition. If the loo breaks before the latest checkpoint save then it reverts to pprevious eval_freq based checkpoint.
- have to modeify the actual train.py file (in lerobot; that the Lehome lerobot-train will use) to incorporate the logic of Karpathy's autoResearch

# BTOP Skill
- build fully grounded skill (all code snippets and directives have traceable sample script collected from docs, turtorial or other open-source repo)
- use the exa Browser and get_code_context skill[in the docs in exa.ai] (for deep semantically related browser serach of related coding and repo related queries and relevant additional context collection)
- use the official huggingface MCP and DeepWiki MCP for deep repo related queries deep understanding of packages, data flows, data types, parameters etc.

# Sweep Script
- Set augmentations to false => the augmentation in backbone variant isn't required for just analysis and you can save significant time.
- had error where "lerobot_policy_dino" existed and impirt worked but it imported from an empty folder => because we did not do a [build-system] in the pyproject.toml (not the universal one) file in the relevenat vision backbone folder + prioritize the src/ directory int he bootstrapper file
- ~~more info regarded bugs faced in the sweep_infra_crash_log.md file.~~

# "Bring Your Own Policy" (BYOP) Scripts for DINOvs and CLIP
- The Eval/Validation Crash: When the custom logic reaches 10,000 steps for the custom backbone, it moves from forward pass the function of forward batch to. Select Actions batch, but the LeRobot implementation for diffusion policy has generate_actions with two keyword parameters for batch and noise. Since initial custom logic for generate_actions only accept (batch), all the training would be lost during the last step of evaluation due to TypeError -> change generate_actions signature to (self, batch, noise=None, **kwargs); set noise which we can set to none thereby meeting the keyword parameters.
- subclassed DiffusionPolicy (which natively just yields self.diffusion.parameters())-> handing the optimizer the entire model, including the heavily frozen DINO and CLIP vision backbones. Most optimizers (like AdamW with weight decay decoupling) will instantly crash or throw memory assertion errors if handed parameters where requires_grad=False: SO MANUALLY OVERRIDE get_optim_param.
- (IMP also for improving speed) Images resized to 224×224 before the DINOv2 and CLIP backbone to avoid processing 480×640 frames (would be extremely slow and OOM-risky) => downsize using bilinear interpolation with antialias.
- Encode a (B*N, C, H, W) image tensor with the frozen DINOv2 backbone.
- Had a few error with key parameter missing between LeRobot 0.4.3 (introduced some new arguments) and the lerobot on the vm.
- U-Net is effective ue to temporal continuity, ultimodal handling and denoising engine (refine from noice to clean trajectory)
- NOTE; Summary of all the errors faced any why i did in detail in the BYOP Crash Summary artifact under Custom_policy folder.

# BYPASSING THE "RESNET ONLY" LeRobot Architecture
- register using the @PreTrainedConfig.register_subclass("...") - understand the new policy names.
- skip validation: use DinoDiffusionConfig and ClipDiffusionConfig to override _post_init_ - > overcome the check for RESNET name in backbone.
-- NOTE IMP: copy-pasting the remaining necessary mathematical validations (handling down-sizing math and noise scheduler assertions)
- skip torchvision: manually map the Normalize/Unnormalize as we skip the super().__init__() so not initialise a torchvision RESnet (skipped the standard Diffusion setup)
- implement get_optim_params(self), which returns only parameters where requires_grad == True -> need as frozen backbone can save VRAM, doin this make sure the optimiser doesn;t try to push graients in to the frozen layers.

# Theorethical underpinnings and 'Gotchas' of CLIP and DINOv2
- Both approaches are vision transformers but they differ in how they handle positional geometries and embeddings.
- CLIP has strictly fixed positional embeddings of 196 (14 into 14 grid -> so 16 into 16 pixels patches in a 224x224 image). ANY more than "No.196" embedding (when larger images are devided using 16x16 px patches) would lead to a hard crash; clip architecture wouldn't know how to handle the remaining and embeddings.
- DINOv2 uses interpolated positional embeddings; can handle higher number of patches even through originally trained on which is the 224x224 images. MAIN issue is downstream -> massive sequence length after flattening and adding to sensor states which would significantly reduce the training speed and cause OOM. 
- CLIP output is pooler.outputs (when cls true - after being sent through contrastive proj layer that align with textual space), DINOv2 output is last_hidden_state[:,0,:]
- NOTE; CLIP trained contrastively (image to text) while DINO is self-supervised (image to image variant): WHAT DOES THIS EXACTLY MEAN
- when doing global_conditioining(), CLIP has a denser feature dim (768) so the single_step_dim formula (robot_state[0] + feature_dim * num_images) becomes substantially larger!! -> Results in dynamic scaling of global_cond_dim of Diffusion U-Net (1-D CNN U-net)
-- CLIP U-net is heavier than DINOv2 U-net
-FUTURE CHANGE for DINOv2; Will need the DINO flatten i have now as it results in a significant loss of the geometric representation as we are collapsing DINOv2's full output — which is (B*N, 257, 384) (256 patch tokens + 1 CLS token) in to a down to a single vector of 384 dimensions
-- - Adding a patch grid reshape + a small trainable SpatialSoftmax head on top of the frozen DINO patch tokens would give you the best of both worlds: DINO's world-model semantics AND explicit geometric keypoints. 

# Garment Classifier Scripts (not trained yet - just the script so far adapted from the resourecs we have)
- lerobot_train_with_plugins.py (The Bootstrapper) : What this does is essentially enable LeRobot framework to correctly see the custom dino policy. and how it actually does is it imports the main from lay robot directly but it does it after manual plugin registration using @register_subclass -> inject the the new policy type in the existing model poll that the original lerobot pulls from.
-- INject vectors: src/ directories of lerobot_policy_dino and lerobot_policy_clip into sys.path.
- modeling_dino_diffusion.py (The Custom Model) : Implement DinoDiffusionPolicy which is a subclass of the standard DiffusionPolicy in the repo, uses facebook/dinov2-small as a fixed feature extractor. By setting requires_grad = False, encode images to lower features. 
-- the DinoDiffusionConfig -> override the _post_init_ that requires a backbone check for RESnet.
- export_garment_classifier_dataset.py (The Extractor) : Build the image set for training the classifier
- train_garment_classifier.py (The Trainer): 2 training loop. (First pass use the pretrained RESNET)
-- first fix the vision backbone and train on the inage dataset to match between (0-3) for 4 cloth categories.
-- 2nd: unfreeze and the train AT LOW LEARNING RATE (1e-4) to sharpen the features.
- sanity check script to help with checking if all the frames are correct.


# DINO and CLIP similarities and benefits
- we need to create a plugin bootstrap to import the vision backbones (DINO and Clip) into the LeRobot framework and create the constructor for the architecture plugin using the existing parameters in the config file. (we can use the existing ResNet18 plugin as a template).
- The DinoV2 captures the intrinsic qualities in the image like depth and spatial features (geometric dictionary) significantly better while the CLIP backbone is better when it comes to semantic extraction or understanding the semantics of the category much better.
- this is wrapper for then lerobot-train.py ; manually trigger existing @register_subclass decorator to prevent discovery issues.
- USe frozen backbone - this way you only used these massive models for BETTER feature extraction and not meant to actively be learned upon or improved with data data. 
- esesentially decouple vision from physics - the Backbone handles the vision and understandiing of what is seen and the U-net learns the physics and Policy.
- DP uses u-Net as the foundation of the classical DP policy; encoder, Bottlenecy and decoder. U-NET in DP also has skip connections to directly connect the encoder and decoder layers to preserve spatial information.
- CNN/U-NET using 1D convolution across time-steps; U-NET is a pysical hardare to choose from.


# DINOv2
- Massive dictionary of visual features: like folds, shadow, wrinkles; help in the policy understand what it sees better.
- better resilienct to lighting and pose variations
- Provide: "Representational Stability: A frozen DINOv2 backbone provides a "high-bandwidth, low-noise" signal of the garment's state, allowing the Diffusion U-Net to converge faster"
- frozen to prevent catastrophic forgetting; the robotic dataset is very small and can have OUTSIZED impact on the model weights that was trained for weeks on supercomputers.
- freezing saves 60-70% memory due to no new gradients
- DinoV2 can be faster on the CPU because we are not operating on the full 480 X something pixels of the input data set rather we sample it down to 224 x 224 (need to confirm) - we are using significantly less compute to actually do the analysis but still getting comparable and even better performance. 

# execution and GPU bugs
- sometimes there can be stray memeory locked in to the GPU; use "kill -9 <PID>"; some these proces are not explicitly shows due  to some hidden away process -> to see all the processes use "ps aux | grep python"
- 

# building vision classifier
- MUST TRAIN the classifier in the image frame collected from all the episodes WITH AUGMENTATIONS - take streams of the top_rgb camera frame - because from here best view of the full cloth we will be handling
- no need for temporal or other metadata: just take the frame(unobstructed view of the cloth) and convert from .paraquet to pyTorch ImageFolder format
- add labels so that it can be used to learn to predict the cloth type from the 4 options. (use 10% validation split as well)
- do sanity check to see of randomly sampled frame to see if the view at the set frame (default 0) is good enough to classify the cloth type.
- ALSO NEED A ROUTER POLICY- to bridge the output from the classifier to dynamically picking the right DP policy; load using Lazy loading (load eachh time to prevent OOM error) or eagar load (DP in VRAM) - Now decide on RESNET.
-- On current observations, it is proposed that it is better to use the DINOv2 for both that way we don't need to have a RESNET (current approach) and a DINOv2 backbone for classification loaded on to the VRAM. -> instead in just one pass you can classify and the route directing in to the U-NET part of the policy without have to sent it through the backbone again
- The VRAM footprint for option A of a unified vision pipeline would be a single DinoV2 (~22M) + one U-NET (~270 M Learnable parameters).; using DINOv2 linear probe as classifier then route th same feature

# Vision Classifier (using existing decorator and tools from lerobot)
- use modelular processorStep (look into it more)
- Before backbone entry, There is a very specific image processing pipeline that needs to be used and there is a specific normalization and dimension permuting as shown (`B, H, W, C` → `B, C, H, W`) ; (Not quite sure what it is - ANS; I belive this is the arder of path grid => B, feature characteristics, ht, width, I THINK???).
- According to NVIDIA, ETH Zurich, the most common cause of classifier failure. in factory floors is due to variable lighting and motion blur. -> Therefore, we did use image augmentation during the training process.
- for the initial version before the sweep, classifier we were going to use (and scripted for) wasresnet simply because of the existing support for RESNET pre-trained models in the LeRobot framework.
- After the sweep analysis,  then we are going to rework it and build a DINO2 variant.

# LeHome Evaluation
- The hackathon evaluator expects a single policy. So we can't send it like a custom script or anything to dynamically choose the pointer to the policies. based on our initial classification.
- So workaround this is to actually create a custom policy that impersonates a policy within it where it would do the same thing of using a classifier to find a share and then load it through its config files Okay. to the right DP policy.
- Depending on VRAM, there are two approaches. 
-- First is lazy loading where at the beginning of each episode you spend a little bit of extra time loading the right policy on to the GPU, VRAM or,
-- it is eager loading where you load all the four DP policies if you have sufficient VRAM and then choose on the fly at the starting of each episode. Which DP to use and since it's already on the vram it doesn't take a lot of time in the beginning of an episode.

# Sweep Updates
- The ResNet variant is almost two times longer for each step which means compared to the other two approaches you could take up to eight hours more. based on the analysis. 
- ResNet does have a lower loss at 10k steps but this could very well be Due to overfitting because ResNet learns everything from scratch other than the frozen backbone and since we have only just started training it could be very well that it's overfitted. 
```
ResNet18: 0.031
DINOv2: 0.037
CLIP: 0.038
```
- Dinov2 and CLIP have very similar loss trajectories and they come with the added benefit of significantly better dense representations that converge beautifully with diffusion policies u-net architecture .
- 
- DINOv2 strikes the perfect balance. Its frozen representation prevents representation collapse, its patch-level attention is perfectly suited for deformable object manipulation, and its compute footprint is vastly superior to training ResNet end-to-end.
