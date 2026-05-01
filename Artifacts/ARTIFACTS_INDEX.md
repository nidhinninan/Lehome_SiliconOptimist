# Artifacts — comprehensive index

Central inventory of everything under [`Artifacts/`](.) (research notes, training readouts, logs, and supporting text). **Last full pass:** 2026-04-30. Regenerate or extend this file when you add new top-level folders or many new documents.

For operational code and VM sync, see also [`../lehome_workspace/lehome_change_log.md`](../lehome_workspace/lehome_change_log.md) and [`../lehome_workspace/vm_transfer_list.md`](../lehome_workspace/vm_transfer_list.md). Cursor plans may live under [`.cursor/plans/`](../.cursor/plans/) and are not duplicated here unless copied into `Artifacts/`.

---

## Root (`Artifacts/*.md`)

| File | Summary |
|------|---------|
| [`act_training_analysis-parameters+config.md`](act_training_analysis-parameters+config.md) | ACT config readout: chunking, VAE/KL, encoder/decoder, batch size. |
| [`first_frame_validity_report.md`](first_frame_validity_report.md) | Frame 0 vs temporal pooling; `top_rgb` for garment view; sanity sampling. |
| [`folding_cloth_research_v1.md`](folding_cloth_research_v1.md) | Cloth IL landscape, LeHome dataset, subvariants, router/VLA/physics tips (v1). |
| [`folding_cloth_research_v2.md`](folding_cloth_research_v2.md) | Same themes as v1, tighter dataset frame counts. |
| [`gpu_profitability_report.md`](gpu_profitability_report.md) | GPU tier economics, DP/X-VLA VRAM and epoch cost tables. |
| [`gpu_report_audit_v2.md`](gpu_report_audit_v2.md) | Independent audit of GPU report (throughput, AMP, frame counts). |
| [`installation_guide.md`](installation_guide.md) | Principia VM: HF, `uv`, IsaacLab, datasets, train/eval, troubleshooting. |
| [`LEARNING-ANALYSIS_context_skill_audit.md`](LEARNING-ANALYSIS_context_skill_audit.md) | DeepWiki vs Exa for implementation-plan quality (dataset aug, etc.). |
| [`Training_extension_CLAUDE.md`](Training_extension_CLAUDE.md) | DP training dynamics: cosine LR vs 30k steps, extend to 100k–150k. |

---

## [`classifiers/`](classifiers/)

| File | Summary |
|------|---------|
| [`dinov2_classifier_training_and_router_requirements_2026-04-29.md`](classifiers/dinov2_classifier_training_and_router_requirements_2026-04-29.md) | DINOv2-based garment classifier + router requirements (post LFM router research). |

---

## [`Custom_policy/`](Custom_policy/)

| File | Summary |
|------|---------|
| [`BYOP-Skill_audit.md`](Custom_policy/BYOP-Skill_audit.md) | Audit of `lehome-lerobot-custom-policy` skill; BYOP vs eval `BasePolicy` path. |
| [`BYOP_training_crash_log.md`](Custom_policy/BYOP_training_crash_log.md) | DINO/CLIP BYOP crashes: registry, namespace pkg, `dataset_meta`, `noise`, optimizer, aug. |
| [`dp_strategy_guide.md`](Custom_policy/dp_strategy_guide.md) | Unified DP vs classifier router vs VLA for hidden-category eval. |
| [`dppo_feasibility_analysis.md`](Custom_policy/dppo_feasibility_analysis.md) | DPPO vs BC; dense reward in sim; aug resume; pretrained ResNet caveats. |
| [`xvla_bimanual_integration_codemap.md`](Custom_policy/xvla_bimanual_integration_codemap.md) | XVLA bimanual codemap (`so101_bimanual`, processors, flow matching). |

---

## [`Dataset-Viz_analysis/`](Dataset-Viz_analysis/)

| File | Summary |
|------|---------|
| [`dataset_details.md`](Dataset-Viz_analysis/dataset_details.md) | Merged HF dataset layout, cameras, normalization, subvariant eval gotcha. |
| [`datasetViz-lerobot-audit_report.md`](Dataset-Viz_analysis/datasetViz-lerobot-audit_report.md) | Audit: `lerobot-dataset-viz` vs `lerobot-edit-dataset` `--root` semantics, conda/ffmpeg. |
| [`datasetViz-lerobot_implementation_plan.md`](Dataset-Viz_analysis/datasetViz-lerobot_implementation_plan.md) | Custom `LeRobotDataset`, `observation.images.*`, `ImageTransforms`, geometric consistency. |

---

## [`Docker_build/`](Docker_build/)

| File | Summary |
|------|---------|
| [`docker_build_blocker_report.md`](Docker_build/docker_build_blocker_report.md) | Docker build blockers and mitigations. |
| [`docker_submission_local_replication_2026-04-30.md`](Docker_build/docker_submission_local_replication_2026-04-30.md) | Local replication of submission Docker flow (audit + implementation notes). |
| [`docker_submission_plan.md`](Docker_build/docker_submission_plan.md) | Submission image plan (W&B artifact, HF registry, no secrets in image). |
| [`dino_docker_policy_integration_notes.md`](Docker_build/dino_docker_policy_integration_notes.md) | DINO BYOP inside Docker submission image. |

---

## [`Evaluations/`](Evaluations/)

| File | Summary |
|------|---------|
| [`eval_speedup_analysis.md`](Evaluations/eval_speedup_analysis.md) | Eval throughput: `--step_hz 0`, CPU sim, parallel eval, avoid `use_fabric`. |

---

## [`Instance_adaptation/`](Instance_adaptation/)

| File | Summary |
|------|---------|
| [`principia_vm_troubleshooting.md`](Instance_adaptation/principia_vm_troubleshooting.md) | Principia / VM environment troubleshooting. |
| [`technical_grounding_architectural_synthesis.md`](Instance_adaptation/technical_grounding_architectural_synthesis.md) | Instance adaptation architecture synthesis. |

---

## [`Optimisation/`](Optimisation/) (training throughput, MAP, merged runs)

| File | Summary |
|------|---------|
| [`150k-run_recommendations.md`](Optimisation/150k-run_recommendations.md) | Recommendations for full 150k-style training runs. |
| [`config_changes_for_fullRun.md`](Optimisation/config_changes_for_fullRun.md) | Config knobs for full runs; frozen DINO rationale cross-link. |
| [`dino_map_registers_oom_eval_resume_curriculum_thread_2026-04-28.md`](Optimisation/dino_map_registers_oom_eval_resume_curriculum_thread_2026-04-28.md) | MAP+registers OOM, eval, resume, curriculum thread. |
| [`dino_map_registers_utilization_wall_clock_synthesis_2026-04-27.md`](Optimisation/dino_map_registers_utilization_wall_clock_synthesis_2026-04-27.md) | GPU wall-clock, `updt_s` vs `data_s`, MAP VRAM, artifact cross-links. |
| [`MAP_head_choice.md`](Optimisation/MAP_head_choice.md) | Rationale for `map_num_queries` (e.g. 8) on multi-garment policy. |
| [`merged_dataset_dino_map_registers_training_discussion_2026-04-28.md`](Optimisation/merged_dataset_dino_map_registers_training_discussion_2026-04-28.md) | Merged-dataset + DINO MAP/registers training discussion. |
| [`training_eval_paths_rclone_phase_plan_discussion_2026-04-28.md`](Optimisation/training_eval_paths_rclone_phase_plan_discussion_2026-04-28.md) | Training/eval paths and rclone phased plan. |

### [`Optimisation/A100/`](Optimisation/A100/)

| File | Summary |
|------|---------|
| [`run_train_dino_merged_a100_implementation_readout_2026-04-28.md`](Optimisation/A100/run_train_dino_merged_a100_implementation_readout_2026-04-28.md) | A100 merged-dino train run readout. |
| [`training_tuning_merged_dataset_synthesis_2026-04-28.md`](Optimisation/A100/training_tuning_merged_dataset_synthesis_2026-04-28.md) | Merged-dataset tuning synthesis (A100 context). |

### [`Optimisation/sweep/`](Optimisation/sweep/)

| File | Summary |
|------|---------|
| [`10k_micro_sweep_implementation_plan.md`](Optimisation/sweep/10k_micro_sweep_implementation_plan.md) | 10k-step micro-sweep BYOP + bash orchestration plan. |
| [`10k_sweep_analysis.md`](Optimisation/sweep/10k_sweep_analysis.md) | Post–10k-step log analysis (ResNet vs DINO vs CLIP). |
| [`10k_sweep_audit_report.md`](Optimisation/sweep/10k_sweep_audit_report.md) | Code audit vs LeRobot 0.4.x (configs, CLI, imports). |
| [`controlledSweep_v2_conversation_chain.md`](Optimisation/sweep/controlledSweep_v2_conversation_chain.md) | Finalized **controlledSweep_v2** A/B/C/D conversation. |
| [`dino_sweep_video_backends_conversation_summary_2026-04-26.md`](Optimisation/sweep/dino_sweep_video_backends_conversation_summary_2026-04-26.md) | DINO sweep + video backend discussion summary. |

---

## [`Readouts/`](Readouts/) (raw logs — not prose guides)

| File | Summary |
|------|---------|
| [`30K_training_readout`](Readouts/30K_training_readout) | Early DP training log (30k context). |
| [`CLIP_Base-10000.log`](Readouts/CLIP_Base-10000.log) | 10k sweep: CLIP backbone. |
| [`DINOv2_Small-10000.log`](Readouts/DINOv2_Small-10000.log) | 10k sweep: DINOv2-Small. |
| [`dp_top_short_training_150K_mid.log`](Readouts/dp_top_short_training_150K_mid.log) | DP top_short mid-150k training log. |
| [`dp_top_short_training_93of150.log`](Readouts/dp_top_short_training_93of150.log) | DP top_short ~93k training log. |
| [`ResNet18_ImageNet-10000.log`](Readouts/ResNet18_ImageNet-10000.log) | 10k sweep: ResNet18. |
| [`terminal_eval`](Readouts/terminal_eval) | Eval terminal trace (success checks, failures). |

---

## [`shell_cmds/`](shell_cmds/)

| File | Summary |
|------|---------|
| [`vast_26apr.txt`](shell_cmds/vast_26apr.txt) | Shell session / commands (Vast instance, 2026-04-26). |

---

## [`Training_Update/`](Training_Update/)

| File | Summary |
|------|---------|
| [`curriculum_augmentation_eval_success_technical_synthesis_2026-04-28.md`](Training_Update/curriculum_augmentation_eval_success_technical_synthesis_2026-04-28.md) | Curriculum + aug + eval success synthesis. |
| [`training-DP_augmentation_resume_strategy_150k.md`](Training_Update/training-DP_augmentation_resume_strategy_150k.md) | Resume with image aug from ~93k toward 150k; loss bump expectation. |
| [`training-DP_progress_analysis_33koutof150k.md`](Training_Update/training-DP_progress_analysis_33koutof150k.md) | DP loss/LR trajectory ~33k of 150k. |
| [`training-DP_progress_analysis_93koutof150k.md`](Training_Update/training-DP_progress_analysis_93koutof150k.md) | DP progress including ~93k checkpoint update. |

---

## [`vision-classifier/`](vision-classifier/)

| File | Summary |
|------|---------|
| [`2026-04-13_conversation_visual_classifier_brainstorm_and_plan.md`](vision-classifier/2026-04-13_conversation_visual_classifier_brainstorm_and_plan.md) | Full transcript: ResNet18 router, lazy load, scripts, `ClassifierRouterPolicy`, plan paths. |
| [`lfm_vision_router_preliminary_research.md`](vision-classifier/lfm_vision_router_preliminary_research.md) | LFM / vision-router preliminary research. |
| [`Vision-classifier_research.md`](vision-classifier/Vision-classifier_research.md) | Garment classifier research: backbones, aug, unified DINO pipeline idea. |
| [`vision_classifier_implementation_plan.md`](vision-classifier/vision_classifier_implementation_plan.md) | Implementation plan: export, train, router JSON, eval integration. |

---

## [`Vision-Backbone/`](Vision-Backbone/) (DINO / CLIP / ResNet BYOP, MAP, autoresearch)

| File | Summary |
|------|---------|
| [`Adapted_lerobot_AutoResearch_implementation_plan.md`](Vision-Backbone/Adapted_lerobot_AutoResearch_implementation_plan.md) | Early autoresearch → internal `lerobot_train` patch plan. |
| [`AutoTimeSweep_vs_10kRun_responses.md`](Vision-Backbone/AutoTimeSweep_vs_10kRun_responses.md) | Karpathy time-budget vs fixed-step 10k sweep debate + transcript. |
| [`DINOv2_CLIP_ResNet_Architectural_Analysis.md`](Vision-Backbone/DINOv2_CLIP_ResNet_Architectural_Analysis.md) | ViT vs ResNet in BYOP: PE, CLS, `global_cond`, SpatialSoftmax tradeoff. |
| [`DINOv2-MAP_implementation_research.md`](Vision-Backbone/DINOv2-MAP_implementation_research.md) | MAP head + registers executable-style plan (grounded in spatial audit). |
| [`dinov2_spatial_pooling_audit_thread_2026-04-25.md`](Vision-Backbone/dinov2_spatial_pooling_audit_thread_2026-04-25.md) | Spatial readout audit: MAP vs SpatialSoftmax; register artifacts. |
| [`Frozen_backbone.md`](Vision-Backbone/Frozen_backbone.md) | Why freeze DINO for IL. |
| [`grounded_autoresearch_analysis.md`](Vision-Backbone/grounded_autoresearch_analysis.md) | Autoresearch timer + “SR per minute” framing. |
| [`grounded_autoresearch_UPDATED_implementation_plan.md`](Vision-Backbone/grounded_autoresearch_UPDATED_implementation_plan.md) | Discard subprocess kill; internal time budget. |
| [`sweep_infra_crash_log.md`](Vision-Backbone/sweep_infra_crash_log.md) | 10k sweep VM/CLI/registry/dataloader failure log + fixes. |
| [`time_injection_audit_Adapt-AutoResearch.md`](Vision-Backbone/time_injection_audit_Adapt-AutoResearch.md) | Why external time wrapper fails LeRobot final eval. |
| [`vision-backbone-research.md`](Vision-Backbone/vision-backbone-research.md) | ResNet vs BYOP ViTs; A4000 speeds; **controlledSweep_v2** table. |
| [`vision_backbone_implementation_plan.md`](Vision-Backbone/vision_backbone_implementation_plan.md) | Vision sweep / throughput wrapper plans (evolved alongside audits). |

---

## Related repo docs (outside `Artifacts/`)

| Path | Role |
|------|------|
| [`Learning.md`](../Learning.md) | Running project notes (eval, classifier option, parallel eval). |
| [`LEARNING-2.md`](../LEARNING-2.md) | Deeper learning log; garment classifier extractor/trainer notes. |
| [`docs/superpowers/specs/2026-04-11-garment-classifier-training-plan.md`](../docs/superpowers/specs/2026-04-11-garment-classifier-training-plan.md) | Pointer to Cursor plan for garment classifier (if present). |

---

## Maintenance

When you add a new **folder** under `Artifacts/` or more than a few files, update this index: append rows under the right section, adjust “Last full pass” date, and keep [`../AGENTS.md`](../AGENTS.md) pointing here.
