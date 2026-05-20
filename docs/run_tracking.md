# Run Tracking

Record enough metadata to compare dynamic loss weighting and self-consistency experiments without relying on local shell history.

## Training Metadata

- Git commit and branch.
- Base model name or local checkpoint path.
- Dataset split revision and whether MTL mode was enabled.
- NTL and EMO weights, including whether negative values were used for dynamic weighting.
- Loss type, training ratio, device id, and resume checkpoint path.
- CUDA version, GPU type, and Python environment.

## Output Bundle

Keep these files together for each run:

- Full command line.
- Training log.
- Best checkpoint or adapter path.
- Inference CSV output.
- Evaluation metrics from `evaluate_module.py`.
- Notes on interrupted or resumed runs.

This keeps CE, NTL, EMO, and dynamic-weighting variants traceable when comparing QWK and error metrics.
