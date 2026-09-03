# Single-sensor outage imputer

Runtime package for **single-sensor outage recovery**: reconstruct one target tag
when it is unavailable while auxiliary sensors remain online.

The model is a **target-conditional denoising regressor (TCDR)**:

```text
input  = [x ; t_tilde ; m_t]
output = t_hat   (scalar)
```

At deployment, outage mode is forced as:

```text
[t_tilde, m_t] = [0, 1]   (0 in standardized target space = train mean)
t_hat = f_theta(x, 0, 1)
```

This repository is the **integration/runtime** surface. The research manuscript,
controlled MCAR/MAR/MNAR validation, and baselines live in a separate lab repo.

## Install

```bash
git clone https://github.com/romulobrito/single-sensor-outage-imputer.git
cd single-sensor-outage-imputer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

HDF5 training needs PyTables:

```bash
pip install -e ".[hdf5,dev]"
```

## Quick start (synthetic smoke)

```bash
python examples/smoke_synthetic.py
```

This fits scalers on synthetic data, trains a tiny TCDR, writes a bundle under
`examples/synthetic_bundle/`, reloads it, and runs `predict`.

## Train from scratch on local industrial data

Keep the historian file **outside git** (or under ignored `data/`). Example:

```bash
python examples/train_from_h5.py \
  --data /path/to/sulfatos_dados_concatenados_formatados_2021.h5 \
  --target 1251_FIT_801C_2 \
  --bundle-dir artifacts/bundle_target_a \
  --hdf-key optional_table_name
```

Omit `--cpu` to use CUDA when available. `--hdf-key` is only required when the
HDF5 file contains more than one table.

This runs blocked chronological split with adaptive head-skip, train-only
feature screening, mask-weighted TCDR training, bundle export, and held-out
outage-mode metrics via `VirtualSensor.predict`.

## Ecosystem integration

```python
from ssoi import VirtualSensor

sensor = VirtualSensor.from_bundle("path/to/bundle")
y_hat = sensor.predict(X_dataframe_or_ndarray)
```

### Bundle contract

A bundle directory must contain:

| File | Role |
|------|------|
| `manifest.json` | metadata / schema version |
| `feature_selection_config.json` | ordered auxiliary feature names |
| `train_feature_means.json` | train-only fill values for NaNs in `x` |
| `scaler_X.joblib` | feature scaler fitted on train |
| `scaler_y.joblib` | target scaler fitted on train |
| `model_config.json` | architecture hyperparameters |
| `best_model.pth` | trained weights |

### Input / output

- **Input:** auxiliaries in the feature order from the bundle (DataFrame columns
  or `ndarray` with shape `(n, n_features)`).
- **Output:** `t_hat` in engineering units (inverse-scaled).
- **Missing auxiliaries:** filled with train means, then `0.0` as fallback.
- **Target channel:** always treated as missing at inference (`m_t=1`).

## Not in scope here

- Industrial raw historian dumps
- Paper LaTeX / figure pipelines
- Full Phase-2 mechanism stress tests

Train offline in your environment, publish the **bundle**, and call this package
only for inference (or for lightweight retrain scripts you add later).

## License / authorship

Research context: UFRJ industrial virtual-sensing work. Adapt target names and
feature lists to your plant tags.
