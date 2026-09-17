# Box-first HAIR annotation

This tool bootstraps candidate bounding boxes for an On-Shelf Availability dataset. Generic YOLOE text prompts localize hair-care packages in overlapping shelf tiles; OpenCLIP ranks each tight crop against every enabled SKU's image/text prototypes. It does not produce approved ground truth, train a production model, or calculate final present/absent OSA. Every generated box needs human review.

## Current state

The runtime builder and model-free tests are implemented. The intended real dataset has 89 permanent classes (IDs 0–88 in workbook `Sheet2` row order), 89 product references, and 632 shelf images. This checkout does **not** currently contain a rebuilt `dist/hair_colab_runtime.zip`: the real product-reference source directory is missing. Synthetic packaging tests verify code composition and byte preservation, not the completeness or accuracy of that real dataset.

The builder never renames or modifies original images, reference ZIPs, or workbook inputs. `--product-images` accepts either a flat image directory or a ZIP archive, including macOS-created ZIPs whose Thai UTF-8 names are missing the ZIP UTF-8 flag. References receive readable ASCII filenames such as `class_000_8851932487177_dove-blue-shampoo-conditioner-slim-pack-6-x-360-330-ml.jpg` inside the runtime archive. `sku_manifest.csv` preserves text barcodes and original Thai names alongside English class names. The temporary `Needs Review` class is ID 89; it is not a permanent SKU.

## Setup and operator sequence

Use Python 3.11 or 3.12. A CUDA GPU is strongly recommended for real inference. `requirements.txt` pins `ultralytics==8.4.149` and `open-clip-torch==3.3.0`; Pillow is needed for preview rendering. Install dependencies and run model-free tests first:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest -q
```

On Linux/macOS use `python3.11` and `.venv/bin/python` instead. For Colab Enterprise, follow [colab/README.md](colab/README.md) and import `colab/hair_colab_enterprise.ipynb`. Only runtime storage is used: no mounted cloud drive or storage-account integration. Uploaded and generated files disappear when the runtime is deleted.

Once the verified product references are available, build the real upload archive using the actual inputs (the product references may remain zipped; extract the shelf archive first):

```powershell
.venv\Scripts\python.exe prepare_hair_colab.py `
  --workbook "C:\path\HAIR SKU list (with Account list).xlsx" `
  --product-images "C:\path\HAIR_SKU2_IMAGE1_images.zip" `
  --shelf-images "C:\path\HAIR_shelf_images" `
  --translations colab/sku_name_translations.yaml `
  --metadata-output colab/generated `
  --output dist/hair_colab_runtime.zip
```

The defaults enforce exactly 89 SKU rows and 632 decodable shelf JPGs. The explicit archive allowlist includes the CLI, runtime helper, annotation modules, model-free tests, requirements, config, and notebook; it excludes the original workbook, local environments, Git history, caches, and original Thai filenames. All archive paths are ASCII; copied image bytes are checksum-verified. Existing destinations are protected unless `--overwrite` is explicit. Source aliases, image-source destinations, and `reviewed_labels` components are rejected even with overwrite.

After extraction, run from the `hair_colab` directory:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python yoloe_autolabel.py --config config.yaml --validate-only --require-enabled-count 89
```

The real-data gate must report `Validation successful: 89 SKU(s), 632 image(s)` before inference. Validation downloads no models and writes no predictions. A different enabled-count gate is possible for development fixtures; it does not establish real-data readiness.

Create a separate pilot with `colab_runtime.write_pilot_config(project, max_images=10)`. Run that config with the same `--require-enabled-count 89` gate. The pilot limits **images only**: every enabled SKU participates regardless of legacy `pilot.class_ids` or `pilot.max_skus`. Inspect original images alongside previews, box tightness, misses, wrong identities, geometry rejections, duplicate removals, zero-candidate images, `Needs Review` ratio, and top-score distributions in `pilot_output/raw_predictions/run.json`. Ranking scores are not calibrated probabilities. Adjust references/settings and repeat if needed. Run the separately marked full-run cell only after the pilot is useful.

Package the full results into two archives with `colab_runtime.package_results(project, platform_destination, review_destination)`, then download both before runtime deletion:

- `hair_ultralytics_platform.zip` — upload to Ultralytics Platform. Contains original image bytes, matching numeric YOLO labels, and root `data.yaml`; its deterministic split is 90% train / 10% validation with seed `hair-osa-v1`.
- `hair_annotation_review.zip` — keep beside the reviewer. Contains ranked suggestions, barcode/Thai-name manifest, reference definitions, metadata, previews, review queue, summaries, and provenance. It is not a second Platform import.

## Model downloads and offline fallback

First inference may download `yoloe-26l-seg.pt`, the YOLOE text encoder (YOLOE26 uses `mobileclip2_b.ts`), and OpenCLIP `ViT-B-32` / `laion2b_s34b_b79k` weights. YOLOE class setup may also install the `ultralytics/CLIP` tokenizer dependency from GitHub. See the [official YOLOE documentation](https://docs.ultralytics.com/models/yoloe/). Internet access, accelerator availability, and dependency installation are environment-specific.

If downloads are blocked, obtain approved compatible weights/dependencies through an allowed route. Upload the detector and OpenCLIP checkpoints into the extracted project using the Files pane, set `yoloe.model` and `matching.pretrained` to their runtime-relative paths, then regenerate the pilot config. Make the YOLOE text encoder/cache and tokenizer dependency available too. Uploading only the detector checkpoint does not make text inference fully offline. Official detector aliases are delegated to Ultralytics unchanged; custom paths resolve relative to the config directory. Do not change permanent IDs, registry order, or reference mappings.

## Inputs and controls

Dataset, output, reference-image, and custom-checkpoint paths resolve relative to the config file. The prepared runtime contains:

```text
config.yaml
sku_manifest.csv
reference_prompts.yaml
references/
shelf_images/
hair_annotation/
tests/
```

The manifest columns are `class_id,barcode,brand,sku_name,sku_name_th,enabled`. IDs and barcodes must be unique; barcodes remain text to preserve leading zeros. `sku_name` is the English class/preview name, while `sku_name_th` is traceability metadata. Labels store numeric IDs, never English or Thai name strings. Platform `data.yaml` maps numeric IDs to English names.

Reference YAML contains a `references` list with `class_id`, `image`, and optional `bbox: [x1, y1, x2, y2]` in source pixels. An explicit ROI is cropped before matching; otherwise localization selects up to three reference views and falls back to a full-image crop marked low quality when needed. Every enabled SKU needs a valid decodable reference. Inspect `reference_diagnostics.json` for fallback quality.

`colab/config.yaml` defines recall-first defaults for manual cleanup: broad package prompts, 1024-pixel tiles with 20% overlap, a 0.05 primary detector threshold, and 1280-pixel detector input. Images with fewer than 10 retained candidates per megapixel automatically receive a second localization pass using 640-pixel tiles, 30% overlap, a 0.02 threshold, and relaxed geometry/NMS limits; primary and retry candidates are merged before matching. Matching uses 80% visual / 20% text similarity, score threshold 0.24, and top-two margin threshold 0.02. Low-score or ambiguous candidates keep their boxes and top-three suggestions but receive temporary class 89. Generic localization prompt IDs are not SKU labels. These settings intentionally prefer extra boxes that a reviewer can delete over silent misses.

## Outputs, resume, and safety

```text
output/raw_predictions/
  labels/<image-stem>.txt
  metadata/<image-stem>.json
  previews/<image-stem><source-suffix>
  review_queue.csv
  reference_diagnostics.json
  provenance.json
  summary.csv
  run.json
```

Each YOLO row is `class_id x_center y_center width height`, normalized with six decimal places. Images with no candidates receive empty labels and explicit review diagnostics. Metadata records tight pixel geometry, localization confidence/provenance, accepted/review status, and ranked permanent SKU identities/scores. Per-image localization diagnostics record whether the recall retry ran, why it ran, and the primary, retry, and merged candidate counts. Run totals expose geometry rejections, duplicates, permanent/review assignments, score summaries, errors, counters, and effective settings/paths.

Pilot outputs use `pilot_output`; full outputs use `output`. Labels are completion markers and are finalized after sidecars. Reruns skip completed images only if retained metadata/review queue and stored provenance match. Changing model weights, settings, SKU identities, or references invalidates a resume. Explicit `--overwrite` affects selected raw candidates only, never human-reviewed labels. Do not point any source or output at `reviewed_labels`.

## Platform review before training

Review **every** box, including accepted SKU suggestions, in this order:

1. Reassign `Needs Review` boxes to the correct SKU, or remove false positives.
2. Fix box geometry so each box tightly encloses one product instance.
3. Verify accepted SKU suggestions, especially near-identical variants.
4. Add missed visible products.
5. Ensure temporary class 89 has no remaining boxes.
6. Only then delete class 89 before training the permanent 89-class detector. Never renumber permanent IDs.

Raw suggestions and their deterministic split are not approved training truth or accuracy claims. Export reviewed labels separately; neither the inference tool nor the builder reads, writes, deletes, or regenerates a `reviewed_labels` tree. Only reviewed images/labels should train the first normal detector. Final OSA logic is downstream.

## Limitations and troubleshooting

Near-identical packaging can confuse matching; tiny/occluded products can be missed even with tiles. A weak pilot requires manual review and a deliberate adjust/repeat/stop decision, not automatic acceptance. Investigate run errors and reference quality before scaling. For no CUDA, use validation without inference, then attach a supported GPU or configure a supported device. Repair invalid reference boxes and duplicate shelf stems through an approved data-management step; the tool stops rather than silently colliding. Review Ultralytics licensing separately before production deployment.
