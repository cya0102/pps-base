# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PPS (Pull-Push Learning Scheme) is a PyTorch implementation for weakly supervised temporal video grounding from the AAAI 2024 paper. The system locates events in videos given sentence queries without frame-level annotations.

**Key Components:**
- Gaussian Mixture Proposals for diverse event representation
- Pull-Push Learning Scheme with pulling/pushing losses
- DualTransformer architecture

## Commands

### Training
```bash
cd pps-main/

# ActivityNet Captions dataset
bash script/train_activitynet.sh
# Equivalent: python train.py --config-path config/activitynet/config.json

# Charades-STA dataset
bash script/train_charades.sh
# Equivalent: python train.py --config-path config/charades/config.json
```

### Evaluation
```bash
cd pps-main/

# Original model (paper version)
bash script/eval_activitynet.sh
bash script/eval_charades.sh

# Refactored model (PPS_re)
bash script/eval_activitynet_refact.sh
bash script/eval_charades_refact.sh
```

### Manual Training/Evaluation
```bash
python train.py --config-path <config.json> [--ckpt-path <checkpoint.pt>] [--eval] [--exp-name <name>] [--seed <int>]
```

### NLTK Setup (one-time)
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

## Architecture

All source code is in `pps-main/`. The architecture follows a layered pattern:

**Entry Point:** `train.py` → CLI parsing, orchestrates Runner

**Runner Layer:** `runner/base_runner.py` → Training/evaluation loops, checkpointing, WandB logging

**Model Layer:** `model/pps.py` → Core PPS model
- Gaussian Mixture Proposals (centroid, range, importance)
- Pull-Push Learning Scheme (pulling loss for event capture, pushing loss for diversity)
- DualTransformer (two-stage cross-attention)

**Module Layer:** `model/module/` → Reusable components
- `transformer.py` - DualTransformer implementation
- `mutihead_attention.py` - Multi-head attention (note: filename typo preserved)
- `attentive_pooling.py` - Attentive pooling mechanism
- `positional_embedding.py` - Positional embeddings

**Dataset Layer:** `dataset/` → Data loading
- `base.py` - BaseDataset with common preprocessing
- `activitynet.py` - ActivityNet Captions (C3D features)
- `charades.py` - Charades-STA (I3D features)

**Optimizer:** `model/optimizer/` → Custom Adam optimizer with fairseq LR scheduling

## Configuration

JSON config files in `config/` control all hyperparameters:
- `activitynet/config.json` - Paper model config
- `activitynet/config_refact.json` - Refactored model config
- `charades/config.json` - Charades-STA config
- `charades/config_refact.json` - Refactored Charades config

Config structure:
- `dataset.*` - Data paths, feature dimensions, vocab settings
- `train.*` - Batch size, epochs, optimizer, WandB toggle
- `model.*` - Hidden dimensions, proposal count, DualTransformer layers
- `loss.*` - Pull/push loss weights (alpha_pull, alpha_inter_push, alpha_intra_push)

## Data Setup

Pre-extracted features required (not in repo):
- ActivityNet: C3D features → `data/activitynet/sub_activitynet_v1-3.c3d.hdf5` (from LGI repo)
- Charades-STA: I3D features → `data/charades/i3d_features.hdf5` (from CPL repo)

## Dependencies

- Python 3.10.8, PyTorch 2.0.1, CUDA 11.6, cuDNN 8
- nltk 3.8.1, wandb 0.15.2, h5py 3.8.0, fairseq 0.12.2
- Note: fairseq may install a different PyTorch version - delete it and keep 2.0.1

## Code Conventions

- No type hints or docstrings (research codebase)
- snake_case for functions/variables, CamelCase for classes
- Private methods prefixed with `_` (e.g., `_build_model`, `_train_one_epoch`)
- Factory pattern via `getattr()` for dynamic class instantiation
- Two model variants: "PPS" (paper) and "PPS_re" (refactored, `_refact` suffix in configs)

## Evaluation Metrics

- Rank@k IoU thresholds: 0.1, 0.3, 0.5, 0.7
- Mean IoU (mIoU)
- Dataset-specific selection strategies

## Output Paths

Relative to `pps-main/`:
- Logs: `log/`
- Checkpoints: `checkpoints/`
- WandB: `wandb/`
