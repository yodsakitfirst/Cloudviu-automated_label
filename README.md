# YOLOE-assisted OSA auto-labeler

This repository contains a small visual-prompting tool for bootstrapping **raw candidate bounding boxes** for an On-Shelf Availability (OSA) dataset. It does not produce approved ground truth, train a production model, or calculate final present/absent OSA results. Every generated box must be reviewed and corrected by a person before training the first normal closed-set YOLO detector.

## Current implementation state

The reusable tool, schemas, and offline test suite are included. The repository does **not** currently contain the business workbook, verified SKU selection rule, reference product images, or shelf images. Consequently:

- `sku_manifest.csv` and `reference_prompts.yaml` are intentionally empty templates;
- the one-image model smoke test and 5-SKU/10-image pilot cannot yet run;
- the exact 79-class V1 scope is **blocked**, and no full run should be attempted;
- no SKU may be guessed, dropped, reordered, or renumbered merely to reach 79.

When source data becomes available, document the workbook filename, sheet, relevant columns, exact selection rule, selected-row count, duplicate resolution, and deterministic class-order rule here before enabling a full run.

## Why detection comes first

The eventual OSA layer can reduce reviewed detections to `present = 1` when a SKU is detected at least once and `absent = 0` otherwise. This tool addresses the earlier problem: the first custom detector needs reviewed boxes for every visible target instance, but no trusted detector or 79-class box dataset exists yet. YOLOE suggestions may reduce manual effort; they do not remove manual review.

## Requirements

- Python 3.11 is the supported runtime.
- A CUDA-capable GPU is strongly recommended for real YOLOE inference.
- Ultralytics is pinned to `8.4.149` because the visual-prompt API is version-sensitive.
- The default checkpoint is `yoloe-26l-seg.pt`; only its boxes are exported.

Create an environment locally on Linux/macOS:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python yoloe_autolabel.py --config config.yaml --validate-only
.venv/bin/python yoloe_autolabel.py --config config.yaml
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe yoloe_autolabel.py --config config.yaml --validate-only
.venv\Scripts\python.exe yoloe_autolabel.py --config config.yaml
```

Colab:

```python
!pip install ultralytics==8.4.149 "numpy>=2.0,<3" "pandas>=2.2,<3" "PyYAML>=6.0,<7" "opencv-python>=4.10,<5"
!python yoloe_autolabel.py --config config.yaml --validate-only
!python yoloe_autolabel.py --config config.yaml
```

The same commands apply to Colab Enterprise after placing the repository and data on its filesystem. Dataset, output, and reference-image paths in the configuration are resolved from the directory containing the main `config.yaml`, even when the reference YAML lives in a nested directory. Recognized bare official YOLOE aliases such as `yoloe-26l-seg.pt` are delegated unchanged to Ultralytics; custom checkpoint names and paths, including a bare `custom.pt`, resolve from the main config directory.

## Input layout and schemas

Expected repository-relative inputs:

```text
config.yaml
sku_manifest.csv
reference_prompts.yaml
references/
  product-reference-images...
shelf_images/
  shelf-images...
```

The manifest is the permanent class registry:

```csv
class_id,barcode,brand,sku_name,enabled
0,0012345678905,Example Brand,Example Shampoo 370ml,true
```

`class_id` is a unique non-negative YOLO training ID. `barcode` is the stable business ID and is always read as text so leading zeros survive. Brand and name must be non-empty; `enabled` accepts only `true` or `false` (case-insensitive). Duplicate IDs or barcodes are fatal.

Reference definitions allow multiple views of one SKU:

```yaml
references:
  - class_id: 0
    image: references/000_0012345678905_front.jpg
  - class_id: 0
    image: references/000_0012345678905_side.jpg
    bbox: [12, 8, 492, 995]
```

An omitted `bbox` uses the full image, which should already be a tight product crop. An explicit box is `[x1, y1, x2, y2]` in source pixels. Files must exist and decode, boxes must lie within the source image, and every active class needs at least one valid reference.

`config.yaml` controls source paths, model settings, output behavior, prompt batch size, and pilot selection. An empty `pilot.class_ids` deterministically selects the first enabled IDs, which is useful only for plumbing tests; replace it with five verified difficult classes before evaluating model usefulness.

## Temporary prompts versus permanent classes

YOLOE receives sequential temporary prompt IDs for each reference-canvas batch. They are never written directly as dataset labels. For example:

```text
temporary prompt 0 -> permanent class_id 5
temporary prompt 1 -> permanent class_id 17
temporary prompt 2 -> permanent class_id 42
```

Multiple views of class 5 repeat temporary ID 0. Every inference result is checked against the batch mapping, translated to the permanent ID, and stored with barcode, name, confidence, pixel box, and batch provenance. Unknown prompt IDs are fatal rather than silently corrupting labels.

## Validate before inference

Validation loads no model and writes no canvases or predictions:

```bash
python yoloe_autolabel.py --config config.yaml --validate-only
```

It checks configuration types and positive sizes, manifest identifiers, reference files and boxes, active-class selection, shelf image discovery, duplicate image stems, output safety, and any required enabled count.

The full V1 gate is stricter:

```bash
python yoloe_autolabel.py --config config.yaml --validate-only --require-enabled-count 79
```

Keep this command blocked until a source-backed rule produces exactly 79 unique enabled SKUs and each has a valid reference. The script supports any positive enabled-class count; 79 is a dataset gate, not a hard-coded algorithm limit.

CLI overrides are available for controlled runs:

```bash
python yoloe_autolabel.py --config config.yaml --output output/smoke
python yoloe_autolabel.py --config config.yaml --overwrite
```

`--overwrite` only applies beneath the selected output's `raw_predictions` namespace. Never point `--output` at `reviewed_labels`.

## Pilot workflow

Start with one verified SKU and one representative shelf image using a temporary config outside tracked files. Use repository-absolute input paths because relative paths resolve from the temporary config's directory, set `max_images: 1`, and use a new output root such as `output/smoke`.

Inspect the reference canvas, every five-token YOLO row, normalized coordinates, permanent IDs, JSON counts/confidences, preview dimensions/readability, and source/reference checksums. Re-run without `--overwrite`; the existing label must remain unchanged and the skipped counter must increase.

Only after that smoke test passes, select about five difficult verified SKUs and ten representative shelf images. Include near-identical packaging, size variants, shampoo/conditioner pairs, bundles versus singles, and small products. Review every original/preview pair and tally visible instances, correct boxes, misses, false positives, wrong-SKU assignments, and materially poor boxes.

Record one decision:

- **GO** — candidates reduce manual labeling effort enough to continue.
- **ADJUST AND REPEAT** — tune only reference boxes/views, confidence, image size, or prompt batch size, then repeat the same pilot.
- **STOP** — fine-grained confusion or small-object recall makes YOLOE unhelpful.

Pilot precision and recall are descriptive manual diagnostics, not production-accuracy claims. Do not add tiled inference, tracking, training, a GUI, or automatic acceptance in response to a weak pilot.

## Outputs and overwrite safety

```text
output/
└── raw_predictions/
    ├── labels/<image-stem>.txt
    ├── metadata/<image-stem>.json
    ├── previews/<image-stem><source-suffix>
    ├── prompt_canvases/batch_000.png
    ├── provenance.json
    ├── summary.csv
    └── run.json

reviewed_labels/  # never read, written, deleted, or regenerated by this tool
```

Each label row is `class_id x_center y_center width height` with six decimal places and normalized values in `[0, 1]`. An image with no candidates still receives an empty label and valid metadata. Labels are the completion marker: an existing label causes the whole image to be skipped unless `--overwrite` is explicitly supplied. A resume is accepted only when `provenance.json` matches the effective model/settings, active-class mappings, references, boxes, and reference checksums whenever any existing label will remain. That includes labels outside a selected overwrite subset: change provenance only in a run that selects and overwrites every existing marker. Matching prompt canvases are reused. `--overwrite` invalidates only selected completion markers before changing sidecars so an interrupted overwrite is retried normally. Sidecars are finalized before the label, and text/JSON/image files are atomically replaced.

Metadata retains the source path/dimensions, model and Ultralytics version, device, image size, thresholds, permanent and temporary IDs, barcode/name, confidence, pixel box, and prompt-batch provenance. `run.json` records the command, UTC timestamp, Git state when available, config snapshot, batch mappings, effective resolved input/output/model paths, CLI overrides, counters, explicit per-image error details, provenance fingerprint, and elapsed runtime. Summaries expose zero-detection and low-confidence classes rather than hiding them.

## Human correction and downstream training

Import the original shelf images and matching files from `raw_predictions/labels` into CVAT using its YOLO detection format. Class order must match permanent manifest IDs. Reviewers must delete false positives, correct wrong classes, add missed instances, and repair poor boxes. Export reviewed labels to a separate `reviewed_labels/` tree.

Only the reviewed images and labels should train the first normal 79-class YOLO detector. The raw candidates are audit material, not training truth.

## Known limitations

- Near-identical SKUs may be confused.
- Small products in whole-shelf images may be missed after resizing.
- YOLOE confidence is useful for ranking and audit, not calibrated probability.
- Whole-image inference is the only implemented strategy; tiled inference is out of scope.
- Checkpoint download and CUDA availability vary by environment.
- Ultralytics licensing should be reviewed separately before production deployment.

## Troubleshooting

- **No CUDA / invalid device:** validate on CPU without inference, then use a supported GPU or set a valid configurable device for the pilot.
- **Weight download failure:** obtain the pinned checkpoint through an approved network path and retry; do not substitute an undocumented API.
- **Invalid references:** verify the file decodes and that `[x1,y1,x2,y2]` lies inside the image with positive area.
- **Duplicate stems:** rename sources only through an approved data-management step, or place them in a separately designed unique-output scheme; this tool stops before overwriting.
- **79-SKU ambiguity:** stop the full run. Record the source gap and continue only with a separately verified subset.

Run automated checks without weights or a GPU:

```bash
pytest -q
python yoloe_autolabel.py --help
```
