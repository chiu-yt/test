# AGENTS.md

## Scope
- This file is for the root project at `/home/zyt/code/OpenPCDet`.
- `MOS-main/` is a nested standalone fork with its own `setup.py`, `README.md`, `docs/`, `tools/`, and `environment.yml`.
- Unless a task explicitly targets `MOS-main/`, prefer changing the root project only.

## Agent Rule Files
- No `.cursorrules`, `.cursor/rules/`, or `.github/copilot-instructions.md` files were found.
- The only `.github` file present is `.github/workflows/close_stale_issues.yml`, which is not an agent instruction file.

## Repository Layout
- `pcdet/`: main Python package.
- `pcdet/datasets/`: dataset loaders, augmentors, processors, and dataset prep code.
- `pcdet/models/`: detectors, heads, backbones, ROI heads, and view transforms.
- `pcdet/ops/`: CUDA/C++ extensions such as `iou3d_nms`, `roiaware_pool3d`, `roipoint_pool3d`, `pointnet2`, `bev_pool`, and `ingroup_inds`.
- `pcdet/utils/`: logging, distributed helpers, geometry, box utilities, and training helpers.
- `pcdet/tta_methods/`: root-project test-time adaptation logic.
- `tools/`: entrypoints (`train.py`, `test.py`, `test_tta.py`, `demo.py`), configs, and wrappers.
- `tools/cfgs/`: model and dataset YAML configs.
- `tools/scripts/`: `torchrun`, legacy distributed, and Slurm wrappers.
- `docs/` and `docker/`: documentation and container files.

## Research Context
- This repo is being used for `BEVFusion` multimodal TTA under adverse-condition domain shift; the older night-domain line remains important context but is now a supplementary real-shift case study rather than the main benchmark.
- Read `Experiment Record.md` before making nontrivial TTA changes.
- The main bottleneck is noisy pseudo labels, especially `pedestrian` and `traffic_cone`, more than a total detector-backbone failure.
- The best stable mainline is `F1 + D1.3`: freeze `image_backbone + neck`, and apply depth-aware pseudo-label filtering only to `pedestrian=0.24` and `traffic_cone=0.34`.
- Keep the current non-finite pseudo-label / loss protections enabled unless a task explicitly re-evaluates them; they stabilized the TTA loop and should be treated as part of the working baseline.
- Do not directly port `MOS-main` single-LiDAR aggregation or iter-only bank behavior into root `BEVFusion`; that transfer was negative.
- Treat broad all-class Top-K pseudo-label filtering as a previously negative direction unless a task explicitly revisits it.
- Current adverse-condition benchmark protocol should use **full nuScenes val**; keep the 602-frame night subset only for supplementary real-scene analysis.
- Current fog benchmark findings: `fog s1` is too weak for main-method evaluation, `fog s3` is the main development benchmark, and `fog s5` is the strong-corruption confirmation benchmark.
- Under strong fog, fused dual-modality predictions can underperform single-modality baselines; treat **cross-modal conflict and fused pseudo-label unreliability** as the main working hypothesis for the next method stage.
- For corruption engineering, prefer `3D_Corruptions_AD` as the online corruption implementation source and use `Robo3D` mainly as a benchmark/layout reference.
- Do not make checkpoint aggregation the paper center; prioritize **conflict-aware pseudo-label reliability modeling** over additional aggregation complexity or fine-grained threshold retuning.
- Be careful with pseudo-label tensor semantics, `SELF_TRAIN.TAR.LOSS_WEIGHT`, `loss.mean()`, `update_global_step()`, and side-channel / augmentation synchronization.

## Auto-Promoted Findings
<!-- AUTO-AGENTS-PROMOTED:BEGIN -->
- No promoted experiment summaries yet. Use `--promote_to_agents --exp_note "why this run matters"` when a result is stable enough to guide future agents.
<!-- AUTO-AGENTS-PROMOTED:END -->

## Working Directory
- Most runnable commands are meant to be executed from `tools/`.
- `tools/_init_path.py` injects `../` into `sys.path`, so `python train.py ...` and `python test.py ...` assume the current directory is `tools/`.
- If you run from repo root, prefix paths with `tools/` or set `PYTHONPATH` explicitly.

## Build / Install Commands
- Install dependencies: `pip install -r requirements.txt`
- Editable install with CUDA extension build: `python setup.py develop`
- `setup.py` builds the project extensions used under `pcdet/ops/`.
- There is no separate root build system beyond the editable install.

## Training Commands
- Run from `tools/`.
- Single GPU: `python train.py --cfg_file cfgs/<dataset>_models/<model>.yaml`
- Common overrides: `--batch_size <N> --epochs <N> --ckpt <path> --pretrained_model <path> --set KEY VALUE ...`
- Multi-GPU: `sh scripts/torch_train.sh <NUM_GPUS> --cfg_file cfgs/<...>.yaml`
- Legacy distributed: `sh scripts/dist_train.sh <NUM_GPUS> --cfg_file cfgs/<...>.yaml`
- Slurm: `sh scripts/slurm_train.sh <PARTITION> <JOB_NAME> <NUM_GPUS> --cfg_file cfgs/<...>.yaml`

## Test / Eval Commands
- The root project does not ship a top-level pytest suite.
- In this repository, "test" usually means model evaluation through `tools/test.py`.
- Single-checkpoint eval: `python test.py --cfg_file cfgs/<dataset>_models/<model>.yaml --ckpt <checkpoint.pth> --batch_size <N>`
- Useful eval flags: `--eval_tag <tag> --save_to_file --infer_time --set KEY VALUE ...`
- Eval all checkpoints in an experiment: `python test.py --cfg_file cfgs/<...>.yaml --eval_all`
- Eval all checkpoints from a specific checkpoint directory: `python test.py --cfg_file cfgs/<...>.yaml --eval_all --ckpt_dir <dir>`
- Multi-GPU eval: `sh scripts/torch_test.sh <NUM_GPUS> --cfg_file cfgs/<...>.yaml --ckpt <checkpoint.pth> --batch_size <N>`
- Legacy distributed eval: `sh scripts/dist_test.sh <NUM_GPUS> --cfg_file cfgs/<...>.yaml --batch_size <N>`
- Slurm eval: `sh scripts/slurm_test_single.sh <PARTITION> --cfg_file cfgs/<...>.yaml --ckpt <checkpoint.pth> --batch_size <N>`
- Slurm multi-GPU eval: `sh scripts/slurm_test_mgpu.sh <PARTITION> <NUM_GPUS> --cfg_file cfgs/<...>.yaml --batch_size <N>`

## Single Test Guidance
- There is no canonical "run one unit test" command in the root project.
- The closest equivalent is a single-checkpoint evaluation: `python test.py --cfg_file cfgs/<...>.yaml --ckpt <checkpoint.pth> --batch_size <N>`
- For inference-path changes, use a smoke test: `python demo.py --cfg_file cfgs/<...>.yaml --data_path <file-or-dir> --ckpt <checkpoint.pth> --ext .bin`
- For dataset-prep changes, validate with the dataset-specific `create_*_infos` entrypoint from `docs/GETTING_STARTED.md`.
- For TTA experiments, consider both `epoch` and `iter` checkpoints; the best result may appear at an early iter rather than the final epoch.

## Dataset Preparation Commands
- KITTI: `python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml`
- NuScenes: `python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml --version v1.0-trainval`
- Waymo: `python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml`
- ONCE: `python -m pcdet.datasets.once.once_dataset --func create_once_infos --cfg_file tools/cfgs/dataset_configs/once_dataset.yaml`
- Lyft: `python -m pcdet.datasets.lyft.lyft_dataset --func create_lyft_infos --cfg_file tools/cfgs/dataset_configs/lyft_dataset.yaml`
- Argoverse2 prep is documented in `docs/GETTING_STARTED.md`.

## Lint / Format Status
- No root `pyproject.toml`, `setup.cfg`, `tox.ini`, `pytest.ini`, `.flake8`, `.pre-commit-config.yaml`, or `ruff.toml` was found.
- There is no documented repo-wide lint command for the root project.
- Do not invent a mandatory formatter or linter step for root changes.
- Preserve surrounding style and run the narrowest meaningful validation command instead.
- `MOS-main/environment.yml` includes `black`, `flake8`, `pytest`, and `yapf`, but that nested project still has no repo-local config for them.

## Python Style Conventions
- Use 4-space indentation.
- Match the surrounding file instead of reformatting unrelated code.
- Imports are usually grouped as standard library, third-party, then local project imports.
- Tool entrypoints often import `_init_path` first, then stdlib, then external packages, then `pcdet` imports.
- Blank lines separate import groups and top-level definitions.
- Type hints are uncommon; add them only when the file already uses them or a new API truly benefits.
- Docstrings and inline comments are sparse; add them only when logic is non-obvious.
- Existing code mixes `%` formatting, `.format(...)`, and f-strings; prefer the local file's current style.
- Long calls are commonly split with hanging indents.

## Naming Conventions
- Use `snake_case` for functions, local variables, and module filenames.
- Use `CamelCase` for classes.
- Use `UPPER_SNAKE_CASE` for constants and most YAML keys.
- Config access is attribute-based, for example `cfg.MODEL`, `cfg.DATA_CONFIG`, and `cfg.OPTIMIZATION`.
- Keep new YAML keys consistent with existing uppercase naming.

## Config Conventions
- `_BASE_CONFIG_` is the standard config inheritance mechanism.
- Keep dataset configs under `tools/cfgs/dataset_configs/` and model configs under `tools/cfgs/<dataset>_models/`.
- Common top-level YAML sections are `CLASS_NAMES`, `DATA_CONFIG`, `MODEL`, and `OPTIMIZATION`.
- Nested module selection is done with `NAME` fields.
- Inline lists and dict literals are common and acceptable.
- Command-line config overrides use `--set KEY VALUE [KEY VALUE ...]` and dotted paths.

## Error Handling And Logging
- Use `common_utils.create_logger(...)` for scripts that need structured logs.
- Train and eval entrypoints log arguments and resolved config; keep that behavior intact.
- The codebase relies heavily on `assert` for invariants and tensor-shape checks.
- Use explicit exceptions for unsupported modes and invalid external inputs.
- `NotImplementedError`, `ValueError`, and `FileNotFoundError` are common.
- Avoid new broad `except:` blocks unless there is a real fallback path.
- Some local modifications use Chinese comments or log lines; preserve the surrounding language/style.

## Paths And Artifacts
- Output directories are usually `output/<exp_group>/<cfg_stem>/<extra_tag>/...`.
- Prefer `pathlib.Path` in entrypoints and filesystem-heavy code.
- Avoid editing generated or local-artifact content unless the task is specifically about it: `build/`, `pcdet.egg-info/`, `.ipynb_checkpoints/`, compiled `.so` files, checkpoints, and training outputs.

## Shell Script Conventions
- Wrappers use `#!/usr/bin/env bash` and `set -x`.
- Uppercase shell variables such as `NGPUS`, `PY_ARGS`, `PARTITION`, and `JOB_NAME` are standard.
- Wrappers assume execution from `tools/` and call `train.py` or `test.py` directly.
- Preserve the current lightweight wrapper style unless a task explicitly asks for shell hardening.

## MOS-main Notes
- `MOS-main/` is a separate project, not a normal subpackage of the root repo.
- It has its own install step `python setup.py develop`, command surface under `MOS-main/tools/`, and docs.
- Root changes should not casually mirror into `MOS-main/`, and vice versa.

## Practical Guidance For Agents
- Prefer small, targeted edits over broad style cleanup.
- Validate with the smallest command that actually exercises the changed path.
- For root-project tasks, do not accidentally mix imports, configs, or commands from `MOS-main/`.
- When documenting commands, mention the expected working directory.
- If a request says "tests", determine whether that means eval, demo inference, dataset prep, or a newly added unit test suite.
