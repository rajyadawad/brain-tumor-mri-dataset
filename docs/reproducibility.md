# Reproducibility

This document records the exact environment, seeds, assumptions, and known sources of
non-determinism so the published metrics and figures can be regenerated and trusted.

## Environment

| Component   | Training (released checkpoints) | CI / published figures |
|-------------|---------------------------------|------------------------|
| TensorFlow  | 2.21 (CPU)                      | 2.19 (CPU)             |
| Keras       | 3.x                             | 3.x                    |
| Python      | 3.10+                           | 3.10 / 3.11            |
| OS          | Windows 11                      | ubuntu-latest          |
| Hardware    | CPU                             | CPU                    |

Any TensorFlow build in the range **2.16 – 2.21** runs both the notebook and the `src/`
package. The supported range is pinned in [`requirements.txt`](../requirements.txt)
(`tensorflow>=2.16,<2.22`). Metrics reproduce to within **≈0.5 points** across this range;
the small drift comes from version-dependent numeric kernels, not from the model.

## Seeds

All RNGs are seeded with `SEED = 42` (see [`src/utils.py`](../src/utils.py) `set_seed()` and the
notebook's imports cell):

```python
os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
numpy.random.seed(42)
tensorflow.random.set_seed(42)
```

This fixes: the `validation_split` carved from `Training/`, augmentation transforms, and weight
initialisation of the from-scratch CNN.

## Configurable paths

Nothing is hard-coded to one machine. Override any of these environment variables:

| Variable                | Default                | Purpose                         |
|-------------------------|------------------------|---------------------------------|
| `BRAIN_MRI_DATA_DIR`    | `<repo>/Dataset`       | root holding `Training/`/`Testing/` |
| `BRAIN_MRI_MODELS_DIR`  | `<repo>/models`        | `.keras` checkpoints            |
| `BRAIN_MRI_RESULTS_DIR` | `<repo>/results`       | metrics + figures output        |

## "Evaluated == released" guarantee (audit items H1–H3)

The two-stage EfficientNet fine-tuning originally reused the Stage-1 callbacks/checkpoint, which
could leave **Stage-1** weights on disk while the **fine-tuned** model was the one evaluated. The
current code fixes this:

1. Each model trains with `EarlyStopping(restore_best_weights=True)` → best-`val_loss` weights are
   left in memory.
2. **Stage 2 uses a fresh `EarlyStopping`** (never the Stage-1 callback), so it cannot compare
   against Stage-1's loss and accidentally keep Stage-1 weights.
3. After training, the in-memory model is **saved**, then **reloaded from disk and re-evaluated**.

Because evaluation runs on the *file on disk*, a green `python -m src.evaluate` proves the released
checkpoint matches the reported metrics.

## Regenerating everything

```bash
# Train from scratch (CPU: tens of minutes to a couple of hours)
python -m src.train

# Evaluate the released checkpoints -> results/metrics/saved_model_eval.json
python -m src.evaluate

# Regenerate every figure embedded in the README
python scripts/generate_assets.py

# Lint + tests
ruff check src tests app
pytest -q
```

## Published metrics (TensorFlow 2.19, CPU)

| Model            | Accuracy | Macro F1 | ECE   | Errors / 1600 |
|------------------|----------|----------|-------|---------------|
| EfficientNetB0   | 0.8531   | 0.8467   | 0.034 | 235           |
| CNN baseline     | 0.7100   | 0.7004   | 0.130 | 464           |

Source of truth: [`results/metrics/saved_model_eval.json`](../results/metrics/saved_model_eval.json).

## Known sources of non-determinism

- **TensorFlow version / CPU kernels** — different builds produce ≈0.1–0.5 pt metric drift even with
  identical seeds and weights. This is why the released-checkpoint numbers (TF 2.21) and the
  CI/figure numbers (TF 2.19) differ slightly.
- **Multithreaded `tf.data`** — input-pipeline ordering under `AUTOTUNE` is not bit-reproducible, but
  the test set is loaded `shuffle=False` so labels always align with predictions.
- **GPU vs CPU** — training on GPU would change results; the released checkpoints are CPU-trained.

## Assumptions

- The Kaggle dataset is downloaded into `Dataset/` (or `BRAIN_MRI_DATA_DIR`) with the official
  `Training/` and `Testing/` sub-folders intact.
- The released `.keras` checkpoints in `models/` are present (they ship with the repo); otherwise run
  `python -m src.train` first.
- Figures in `results/figures/` and `assets/` are generated artifacts — edit the code, not the PNGs.
