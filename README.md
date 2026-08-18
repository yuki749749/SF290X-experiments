# SF290X Experiments

This repository contains the codebase and experiment scripts for the degree project, organized to run the entire pipeline with custom domain sizes and planning horizons.

## Setup

1. Make sure you have `uv` installed.
2. Initialize virtual environment:
   ```bash
   uv venv
   .venv\Scripts\activate
   ```
3. Install package in editable mode with dependencies:
   ```bash
   uv pip install -e .
   ```

## Running the Pipeline

1. **GP Hyperparameter Tuning**:
   ```bash
   python experiments/gp_tuning/gp_tuning.py
   ```
2. **Training Data Generation**:
   ```bash
   python experiments/training_data_generation/generate_raw_trajectories.py -m planner=random,bo scenario_idx="range(0,500)"
   ```
3. **Trajectory Slicing/Post-Processing**:
   ```bash
   python experiments/training_data_generation/process_trajectories.py
   ```
4. **Diffusion Model Training**:
   Update `paths.trajectories.processed` in `config/train_diffusion.yaml` to point to the output processed data path, then run:
   ```bash
   python experiments/train_diffusion/train_diffusion.py
   ```
5. **Evaluation**:
   Update `planner.checkpoint_path` in `config/planner/diffusion.yaml` to point to the trained model checkpoint, then run:
   ```bash
   python experiments/evaluation/evaluation.py
   python experiments/conditioning/conditioning_ablation.py
   ```
