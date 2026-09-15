# Colab Enterprise runtime-local workflow

This workflow processes all 632 shelf images against all 89 enabled HAIR product classes. It uses only the Colab Enterprise runtime filesystem. It does not require Google Drive or Cloud Storage access.

## Before opening the notebook

The prepared files are:

- `colab/hair_colab_enterprise.ipynb` — import this notebook into Colab Enterprise;
- `dist/hair_colab_runtime.zip` — upload this archive through the notebook Files pane.

All paths inside the ZIP are ASCII-safe. Product names remain in Thai inside `sku_manifest.csv` so labels retain the exact business names. Product reference files use names such as `class_000_8851932487177.jpg`.

## Runtime steps

1. Import `hair_colab_enterprise.ipynb` into Colab Enterprise.
2. Connect the notebook to a GPU runtime with enough free disk space.
3. Upload `hair_colab_runtime.zip` through the Files pane.
4. Run the extraction and environment-check cells.
5. Run the dependency-installation and offline-test cells.
6. Run validation. It must report `Validation successful: 89 SKU(s), 632 image(s)`.
7. Run the 5-SKU/10-image pilot and inspect its previews.
8. Run the separately marked full-run cell only after the pilot is acceptable.
9. Run the result-packaging cell.
10. Download `hair_label_results.zip` from the Files pane.

Colab Enterprise deletes uploaded and generated runtime files when the runtime is deleted. Download the result archive before deleting the runtime.

## Model checkpoint

The default configuration uses the official `yoloe-26l-seg.pt` alias. Ultralytics downloads it on first use when the runtime has public internet access.

If public internet access is blocked, upload the checkpoint into the extracted `hair_colab` directory and change only this value in its `config.yaml`:

```yaml
yoloe:
  model: uploaded_checkpoint.pt
```

Do not change class IDs, manifest order, or reference mappings when replacing the checkpoint.

## Pilot and full-run separation

The notebook writes pilot candidates beneath `pilot_output/raw_predictions`. The full run writes beneath `output/raw_predictions`. This prevents pilot completion markers from causing full-run images to be skipped.

Within the same runtime, rerunning the full-run cell without overwrite skips completed labels only when the stored provenance still matches the model, settings, classes, and references. Do not use overwrite unless the existing raw candidates should be regenerated.

## Human review

`hair_label_results.zip` contains raw YOLOE suggestions, metadata, previews, summaries, and provenance. Reviewers must remove false positives, correct wrong classes and boxes, and add missed instances before using the labels to train a normal 89-class YOLO detector. These files are not reviewed ground truth and do not directly calculate final on-shelf availability.
