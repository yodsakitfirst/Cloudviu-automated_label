# Colab Enterprise runtime-local workflow

Generic YOLOE text prompts propose tight boxes in overlapping shelf tiles; OpenCLIP ranks those crops against all enabled HAIR SKU identities. Low-score or ambiguous boxes use temporary class 89, `Needs Review`. These are raw suggestions, never approved training truth or final OSA results.

## Before opening the notebook

Import `colab/hair_colab_enterprise.ipynb`. A rebuilt `dist/hair_colab_runtime.zip` must be prepared from all verified source inputs before upload. This checkout has no rebuilt real-data ZIP yet: the product-reference source directory is missing. The intended dataset is 89 permanent SKUs, 89 product references, and 632 shelf images. Synthetic packaging verification does not establish that real-data gate.

The builder uses an explicit code/test allowlist, ASCII archive paths, and checksum-verified image copies without modifying original inputs. `sku_manifest.csv` maps numeric permanent IDs 0–88 to English `sku_name` values and preserves original Thai `sku_name_th` values and text barcodes. Prepared reference names look like `class_000_8851932487177.jpg`.

## Runtime steps

1. Import the notebook into Colab Enterprise and connect a GPU runtime with enough disk space.
2. Upload the rebuilt `hair_colab_runtime.zip` through the Files pane.
3. Run extraction and the environment check. Paths remain local to the extracted project.
4. Install dependencies with `pip install -r requirements.txt`, then run `pytest -q`. These tests use fake model backends; they require no weights or GPU.
5. Validate with `--validate-only --require-enabled-count 89`. The real package must report `Validation successful: 89 SKU(s), 632 image(s)` before continuing.
6. Create the pilot with `write_pilot_config(project, max_images=10)` and run it. It limits images only: all 89 enabled SKU identities remain available, regardless of legacy class-selection controls.
7. Inspect the preview grid and original images for box tightness, misses, and incorrect SKU suggestions. Read the printed `pilot_output/raw_predictions/run.json`, geometry rejection counts, duplicate removals, zero-candidate images, `Needs Review` ratio, and top-score distributions. Scores rank suggestions; they are not calibrated probabilities. Investigate errors and low-quality reference fallbacks. Adjust and repeat if needed.
8. Run the separately marked full-run cell only after the pilot is acceptable. It processes all 632 images against all 89 enabled classes.
9. Package results with the two-destination `colab_runtime.package_results` call.
10. Download **both** `hair_ultralytics_platform.zip` and `hair_annotation_review.zip` from the Files pane before deleting the runtime.

The notebook uses only the runtime filesystem, without mounted cloud drives or storage-account integration. Uploaded and generated files disappear when the runtime is deleted. Result archives are written beside the notebook workspace, not to a fixed machine-specific path. Existing ZIPs are protected unless `overwrite=True` is explicitly requested.

## First-use downloads and checkpoint fallback

First inference may download `yoloe-26l-seg.pt`, the YOLOE text encoder (`mobileclip2_b.ts` for YOLOE26), and OpenCLIP `ViT-B-32` / `laion2b_s34b_b79k` weights. YOLOE class setup may also install the `ultralytics/CLIP` tokenizer dependency from GitHub. Consult the [official YOLOE documentation](https://docs.ultralytics.com/models/yoloe/). The runtime must allow those downloads or have compatible dependencies/caches already available.

If public network access is blocked, upload approved compatible detector and OpenCLIP checkpoints into the extracted `hair_colab` project using the Files pane. Edit the extracted `config.yaml` before generating the pilot:

```yaml
yoloe:
  model: uploaded_detector.pt
  imgsz: 1280
  device: 0
matching:
  # Keep the other matching controls from config.yaml.
  model: ViT-B-32
  pretrained: uploaded_openclip.pt
```

This is a partial edit example, not a replacement config. Paths resolve relative to `config.yaml`. The YOLOE text encoder/cache and tokenizer dependency must also be available: uploading only the detector is insufficient for fully offline text inference. Do not change permanent IDs, registry order, or reference mappings. Install compatible dependencies through an approved route if pip is also blocked.

## Pilot, resume, and two ZIPs

Pilot candidates live beneath `pilot_output/raw_predictions`; full candidates live beneath `output/raw_predictions`. This separation keeps pilot completion markers from skipping full-run images. Within one runtime, rerunning without overwrite skips completed labels only when stored provenance and retained metadata/review queue match the model, settings, identities, and references. Overwrite regenerates selected raw suggestions only. `reviewed_labels` is never an automated input or output.

`hair_ultralytics_platform.zip` goes to Ultralytics Platform. It has root `data.yaml`, original image bytes, matching numeric YOLO labels, and a deterministic 90% train / 10% validation split (`hair-osa-v1`). `data.yaml` maps permanent numeric labels to English names and appends temporary class 89, `Needs Review`; class names are not written into YOLO rows.

`hair_annotation_review.zip` stays beside the reviewer. It carries top-three ranked suggestions, manifests/barcodes/Thai names, reference definitions, previews, metadata, review queue, run summaries, reference diagnostics, and provenance. Use it as audit/review material, not as a Platform dataset import or approved ground truth.

## Six-step Platform review

Review every box, not just uncertain ones:

1. Reassign `Needs Review` boxes to the correct SKU, or delete false positives.
2. Fix box geometry to tightly enclose one product instance.
3. Verify accepted SKU suggestions, especially similar packages and sizes.
4. Add missed visible products.
5. Confirm class 89 contains no remaining boxes.
6. Only then delete class 89 before training the permanent 89-class detector. Never renumber permanent IDs.

Export reviewed labels separately. Raw candidates are audit material; only corrected, reviewed labels should train the first normal detector.
