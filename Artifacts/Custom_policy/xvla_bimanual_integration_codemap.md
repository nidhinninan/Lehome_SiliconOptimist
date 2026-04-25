# XVLA Policy Integration for Bimanual Cloth Folding

This codemap traces the XVLA policy integration workflow for bimanual cloth folding in Isaac Sim, covering configuration setup [1a-1d], preprocessing pipeline [2a-2d], model inference [3a-3d], and training configuration [4a-4d]. Key integration points include the `so101_bimanual` action mode [1b] and the domain ID preprocessing [2b].

### 1. XVLA Policy Configuration and Action Mode Setup
How XVLA policy is configured for bimanual robots with action mode selection

#### 1a. Policy Type Specification (`xvla.mdx:64`)
Set policy type to XVLA in configuration
```text
policy.type=xvla
```

#### 1b. Bimanual Action Mode Registration (`action_hub.py:428`)
Action mode for 2-arm setup maps 12D real robot to 20D model
```text
@register_action("so101_bimanual")
class BimanualSO101ActionSpace(BaseActionSpace):
```

#### 1c. Action Dimension Mapping (`action_hub.py:443`)
Real robot 12D vs model 20D action dimensions
```text
REAL_DIM = 12
dim_action = 20
```

#### 1d. Action Mode Configuration (`xvla.mdx:165`)
Configure policy to use bimanual action mode
```text
policy.action_mode=so101_bimanual
```

### 2. XVLA Preprocessing Pipeline for Isaac Sim
How observations are preprocessed for XVLA in Isaac Sim environments

#### 2a. Required Preprocessing Steps (`processor_xvla.py:72`)
Essential XVLA preprocessing pipeline components
```text
XVLAImageToFloatProcessorStep(),
XVLAImageNetNormalizeProcessorStep(),
XVLAAddDomainIdProcessorStep()
```

#### 2b. Domain ID Addition (`processor_xvla.py:438`)
Add domain identifier for different robot configurations
```text
def __call__(self, transition: EnvTransition) -> EnvTransition:
    comp["domain_id"] = torch.tensor([int(self.domain_id)] * batch_size, dtype=torch.long)
```

#### 2c. ImageNet Normalization (`processor_xvla.py:408`)
Apply ImageNet statistics to images
```text
obs[key] = (tensor - mean) / std
```

#### 2d. Isaac Sim Environment Loading (`envhub_leisaac.mdx:288`)
Load bimanual cloth folding environment from EnvHub
```text
envs_dict = make_env("LightwheelAI/leisaac_env:envs/bi_so101_fold_cloth.py", n_envs=1, trust_remote_code=True)
```

### 3. XVLA Model Forward Pass and Action Generation
How XVLA processes inputs and generates actions for bimanual control

#### 3a. Model Forward Pass (`modeling_xvla.py:194`)
XVLA model processes multimodal inputs
```text
def forward(
    self,
    input_ids: torch.LongTensor,
    image_input: torch.FloatTensor,
    image_mask: torch.Tensor,
    domain_id: torch.LongTensor,
    proprio: torch.Tensor,
    action: torch.Tensor,
) -> dict[str, torch.Tensor]:
```

#### 3b. Action Space Preprocessing (`modeling_xvla.py:220`)
Apply action mode-specific preprocessing
```text
proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)
```

#### 3c. Action Generation (`modeling_xvla.py:232`)
Generate action sequence using flow matching
```text
def generate_actions(
    self,
    input_ids: torch.LongTensor,
    image_input: torch.FloatTensor,
    image_mask: torch.Tensor,
    domain_id: torch.LongTensor,
    proprio: torch.Tensor,
    steps: int,
) -> torch.Tensor:
```

#### 3d. Action Postprocessing (`modeling_xvla.py:267`)
Apply action mode-specific postprocessing and trim to real robot dimensions
```text
return self.action_space.postprocess(action)
```

### 4. XVLA Training Configuration for Fine-tuning
How to configure XVLA training for new embodiments like bimanual cloth folding

#### 4a. Training Command Setup (`xvla.mdx:156`)
Command line configuration for XVLA fine-tuning
```text
--policy.path="lerobot/xvla-base" \
--policy.action_mode=so101_bimanual \
--policy.dtype=bfloat16 \
--policy.freeze_vision_encoder=false \
--policy.freeze_language_encoder=false
```

#### 4b. Auto Action Space Building (`modeling_xvla.py:67`)
Build action space with automatic dimension detection
```text
self.action_space = build_action_space(
    config.action_mode.lower(),
    real_dim=real_dim,
    max_dim=config.max_action_dim,
)
```

#### 4c. Action Padding Logic (`action_hub.py:468`)
Pad real robot actions to model dimensions
```text
def _pad_to_model_dim(self, x: torch.Tensor) -> torch.Tensor:
    """If last dim is REAL_DIM (12), pad zeros to reach dim_action (20)."""
```

#### 4d. Action Trimming Logic (`action_hub.py:482`)
Trim model outputs to real robot dimensions
```text
def _trim_to_real_dim(self, x: torch.Tensor) -> torch.Tensor:
    """Keep only the first REAL_DIM (12) dims for the real robot."""
```
