# LeRobot v0.4.3 verification references

Minimal copy-paste skeletons for BYOP and eval registration: see **Templates** in [`SKILL.md`](./SKILL.md).

Use these URLs when cross-checking BYOP behavior. The challenge pins `lerobot==0.4.3` ([`pyproject.toml`](../../../lehome_workspace/lehome-challenge/pyproject.toml)).

## Tagged source (GitHub raw, v0.4.3)

- [`src/lerobot/configs/policies.py`](https://raw.githubusercontent.com/huggingface/lerobot/v0.4.3/src/lerobot/configs/policies.py) — `PreTrainedConfig` base (`draccus.ChoiceRegistry`), `from_pretrained`.
- [`src/lerobot/policies/factory.py`](https://raw.githubusercontent.com/huggingface/lerobot/v0.4.3/src/lerobot/policies/factory.py) — `make_policy`, `make_policy_config`, `make_pre_post_processors`, `_get_policy_cls_from_policy_name`, `_make_processors_from_policy_config`.
- [`src/lerobot/policies/diffusion/configuration_diffusion.py`](https://raw.githubusercontent.com/huggingface/lerobot/v0.4.3/src/lerobot/policies/diffusion/configuration_diffusion.py) — example `@PreTrainedConfig.register_subclass("diffusion")` on a config dataclass.

## Verified behaviors (v0.4.3 factory.py)

- **Unknown `policy_type` in `make_policy_config`**: falls through to `PreTrainedConfig.get_choice_class(policy_type)` and instantiates that class—intended for registered subclasses (plugins).
- **`get_policy_class`**: built-in names use explicit imports; otherwise `_get_policy_cls_from_policy_name(name)` requires `name in PreTrainedConfig.get_known_choices()`, derives `*Policy` from `*Config`, imports `modeling_*` from the config module’s sibling naming (`configuration_X` → `modeling_X`).
- **Processors for plugins**: `_make_processors_from_policy_config` imports `make_{config.type}_pre_post_processors` from `processor_*` (`configuration_*` → `processor_*`).

## Official documentation

- [Bring your own policies](https://huggingface.co/docs/lerobot/bring_your_own_policies)
- [LeRobot docs hub](https://huggingface.co/docs/lerobot)

## DeepWiki (secondary)

- [huggingface/lerobot on DeepWiki](https://deepwiki.com/huggingface/lerobot) — useful for navigation; confirm any API detail against v0.4.3 source or HF docs above.

## LeHome eval ↔ LeRobot stack

The challenge’s [`lerobot_policy.py`](../../../lehome_workspace/lehome-challenge/scripts/eval_policy/lerobot_policy.py) imports:

`PreTrainedConfig`, `make_policy`, `make_pre_post_processors`, `LeRobotDatasetMetadata`, `TransitionKey` from the installed `lerobot` package—must match 0.4.3 APIs.
