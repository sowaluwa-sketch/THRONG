# THRONG

THRONG is a relation-aware temporal hypergraph detector for group-level
malicious-crowd detection in encrypted traffic. The released implementation
contains destination recurrence (DR), cross-member template reuse (TR), shift
alignment (SA), dynamic hypergraph propagation, and the GTDA temporal encoder.

## Release contents

```text
code/throng/       Core model, relation extraction, GTDA encoder, and adaptation API
data/Traffic/      Encrypted-session metadata and release-level split files
requirements.txt   Python package constraints
```

The commands below cover environment setup, data inspection, model construction,
and inference through the public API.

## Environment

The code targets Python 3.10 or newer and requires:

```text
numpy==1.26.4
torch>=2.1,<3.0
```

Create an isolated environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/code"             # Windows: set PYTHONPATH=%CD%\\code
```

On Windows, activate the environment with `.venv\\Scripts\\Activate.ps1`
and set `PYTHONPATH` to the absolute `code` directory. CUDA is optional. The
model follows the device of its parameters, so use `model.to("cuda")` when a
CUDA-enabled PyTorch installation is available.

## Model configuration

The default `THRONG()` configuration is the configuration used by the released
core model:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `hidden` | 64 | Member, session, and temporal hidden width |
| `dropout` | 0.1 | Dropout in propagation, fusion, and classifier blocks |
| `bins` | 8 | Number of temporal windows |
| `hypergraph_layers` | 2 | Dynamic relation-propagation layers |
| `relation_dropout` | 0.1 | Training-time relation masking probability |
| `template_cosine_threshold` | 0.75 | TR pair-connection threshold |
| `shift_cosine_threshold` | 0.70 | SA pair-connection threshold (`tau_s`) |
| `shift_minimum_norm` | 0.10 | Minimum shift norm for SA |
| `relation_scale_init` | 0.10 | Initial per-relation propagation scale |
| `relation_context_scale_init` | 0.10 | Initial relation-context scale |
| `temporal_residual_scale_init` | 0.10 | Initial temporal residual scale |
| `temporal_mix_init` | 0.25 | Initial temporal mixing scale |

The input projection sizes are 49 member features, 15 session features, and
four relation-context values. `GroupRecord` is defined in
`code/throng/types.py`. Its session tensors use integer `session_bins`,
`session_users`, and `session_targets` together with floating-point session
features. The model predicts the two classes in the order
`("Other", "Malicious")`.

## Data

The release contains encrypted-session metadata. Payloads, raw addresses, and
direct source identifiers are not model inputs. The TSV files use stable
pseudonymous identifiers and include group labels, sessions, and split
assignments.

| Dataset | Groups | Users | Sessions | Intended use |
| --- | ---: | ---: | ---: | --- |
| Public | 3,340 | 23,316 | 30,956 | Source evaluation |
| CE-A | 4,177 | 33,346 | 220,029 | Controlled source evaluation |
| CE-B | 3,858 | 52,065 | 200,356 | Robustness test only |
| External | 3,041 | 20,663 | 22,495 | Cross-source evaluation |

`Public`, `CE-A`, and `External` provide release-level group split files.
`CE-B` is paired with held-out CE-A tracks and is marked
`robustness_test`. Its manifest records perturbation levels 0.1 through 0.6,
label-independent perturbation, and preservation of task objectives and labels.

## Reproduction commands

Run these commands from the repository root after installing dependencies.

Check the package import and the released default configuration:

```bash
python -c "from throng import THRONG, CLASS_NAMES; m=THRONG(); print(m.hidden, m.bins, CLASS_NAMES)"
```

On Windows PowerShell, use the same command after setting `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "$PWD\\code"
python -c "from throng import THRONG, CLASS_NAMES; m=THRONG(); print(m.hidden, m.bins, CLASS_NAMES)"
```

Inspect the release-level group counts and splits:

```bash
python - <<'PY'
import csv
from pathlib import Path

root = Path("data/Traffic")
for name in ("Public", "CE-A", "CE-B", "External"):
    split_file = root / name / "splits.tsv"
    with split_file.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    counts = {}
    for row in rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    print(name, counts)
PY
```

Construct `GroupRecord` objects from the session TSV files and run the model:

```python
import torch
from throng import THRONG

# `records` must be a list of GroupRecord objects produced by your TSV loader.
model = THRONG(hidden=64, bins=8, hypergraph_layers=2)
model.eval()
with torch.no_grad():
    logits = model.forward_batch(records)
    predictions = logits.argmax(dim=-1)
```

For source-calibrated zero-shot filtering, use
`calibrate_zero_shot_threshold` on source-validation records and pass the
result to `zero_shot_filter`. For few-shot transfer, use
`fit_few_shot_head` on labeled support records and then `few_shot_predict`.

## Data schema and citation

Each dataset directory contains `sessions.tsv`, `group_labels.tsv`, and
`splits.tsv`. The CE-B directory additionally contains pairing and perturbation
audit files. Dataset manifests record row counts and release metadata. Please
cite the accompanying THRONG paper when using the code or data.
