# Colab Enterprise Hair Auto-Label Package Design

## Purpose

Prepare the existing YOLOE-assisted labeling program for a single, runtime-local Colab Enterprise workflow. The package will use 89 hair-product reference images to propose SKU-level bounding boxes for all 632 shelf images. Generated annotations are review candidates, not approved ground truth or final on-shelf-availability results.

## Confirmed Scope

- Keep `yoloe_autolabel.py` as the labeling engine.
- Use the 89 product rows in `Sheet2` of `HAIR SKU list (with Account list).xlsx` as the permanent class registry.
- Run all 89 product classes against every shelf image. Account and supermarket columns do not restrict inference.
- Use only Colab Enterprise runtime storage. Do not mount Google Drive and do not require Cloud Storage permissions.
- Use ASCII-only filesystem names in the prepared package.
- Preserve Thai product names as manifest data.
- Do not rename or modify the supplied source files in `/Users/me/Downloads/Cloudviu-data`.

## Source Data Findings

- `Sheet2` has a header in workbook row 2 and 89 data rows in rows 3 through 91.
- All 89 selected rows have a nonblank, unique barcode and product name.
- Barcodes contain 13 or 14 digits and must be stored as text.
- `HAIR_product_images` contains 89 decodable product images.
- Eighty-eight product-image stems match the corresponding `Sheet2` product names after Unicode and whitespace normalization.
- One source pair differs only by `/` versus `-`: workbook name `เคลียร์แชมพูผช คูล 25บ/29B 12X6X60มล.` and image name `เคลียร์แชมพูผช คูล 25บ-29B 12X6X60มล..png`. This pair will be mapped explicitly rather than by fuzzy matching.
- `HAIR_shelf_images` contains 632 decodable `.jpg` shelf images. Its `.csv`, `.txt`, `.sh`, and `.DS_Store` files are not inference inputs.

## Prepared Package

The deliverable will contain one uploadable archive with this internal layout:

```text
hair_colab/
├── hair_colab_enterprise.ipynb
├── yoloe_autolabel.py
├── requirements.txt
├── tests/
│   └── test_yoloe_autolabel.py
├── config.yaml
├── sku_manifest.csv
├── reference_prompts.yaml
├── references/
│   └── class_<three-digit-class-id>_<barcode>.<source-extension>
├── shelf_images/
│   └── <existing-ASCII-image-name>.jpg
└── output/                    # created by the labeling program
```

The top-level archive will be named `hair_colab_runtime.zip`. Generated inference output will not be included in the input archive.

## Class Registry and Reference Mapping

Class IDs will be assigned as `0` through `88` in the existing `Sheet2` row order. That order will be materialized in `sku_manifest.csv`; it will not be recalculated during inference.

Each manifest row will contain:

- `class_id`: permanent YOLO class ID;
- `barcode`: original workbook value stored as text;
- `brand`: `Unspecified`, because the workbook does not contain a verified brand column and the existing script requires a nonblank value;
- `sku_name`: original Thai product name;
- `enabled`: `true` for all 89 rows.

Each product image will be copied to `references/` without changing its image bytes. Its destination name will use `class_<three-digit-class-id>_<barcode>` plus a lowercase version of the source extension. `reference_prompts.yaml` will map each class ID to exactly one copied reference image and omit `bbox`, causing the existing program to treat the full product image as the visual prompt.

The workbook, original Thai filenames, account-list columns, source download scripts, and source URL files are preparation inputs or audit context. They are not required inside the runtime archive after the manifest and reference mapping are materialized.

## Colab Enterprise Workflow

The notebook will be a thin orchestrator around the existing command-line program:

1. Identify its working directory dynamically instead of assuming `/content`.
2. Verify that a GPU is visible and report the Python version, available disk space, and CUDA status.
3. Install the pinned requirements.
4. Run the existing offline test suite.
5. Run `yoloe_autolabel.py --config config.yaml --validate-only --require-enabled-count 89`.
6. Run a small pilot by using a notebook-created temporary configuration that changes only the pilot settings and output directory.
7. Display pilot previews for human inspection.
8. Run the full configuration only after the user executes the full-run cell.
9. Package `output/raw_predictions` as `hair_label_results.zip` for download through the Colab Enterprise Files pane.

The notebook will explain that runtime files are deleted when the runtime is deleted. It will instruct the user to download `hair_label_results.zip` before deleting the runtime. It will not contain Drive-mount, `gs://`, GCS client, or bucket-sync code.

If the runtime has public internet access, Ultralytics may download the official `yoloe-26l-seg.pt` alias. The configuration will also explain how to replace the alias with an uploaded checkpoint path when network policy blocks checkpoint download.

## Configuration

The runtime `config.yaml` will use only package-relative paths:

- manifest: `sku_manifest.csv`;
- references: `reference_prompts.yaml`;
- shelf images: `shelf_images`;
- output root: `output`;
- official model alias: `yoloe-26l-seg.pt`;
- enabled count: 89, enforced by the validation command.

The existing model and output safety behavior remains unchanged. Pilot output and full output use separate roots so a pilot cannot become a misleading completion marker for the full run.

## Output and Review Boundary

The existing `raw_predictions` structure remains authoritative. It includes YOLO label text files, per-image metadata, previews, prompt canvases, provenance, run metadata, and summaries. Labels preserve shelf-image stems, which are already ASCII-safe.

The package will not:

- mark generated labels as reviewed;
- write to `reviewed_labels`;
- train a closed-set detector;
- infer final product presence or absence;
- modify the original workbook or images;
- translate Thai product names into new business names.

Reviewers must correct wrong classes, false positives, missed products, and poor boxes before the labels are used for training.

## Validation and Failure Handling

Package construction will fail before creating the final archive if any of these conditions occurs:

- selected workbook row count is not 89;
- barcode or product name is missing or duplicated;
- a barcode is not a 13- or 14-digit text identifier;
- a reference match is missing or ambiguous;
- the explicit one-off filename mapping no longer resolves uniquely;
- a product or shelf image cannot be decoded;
- the prepared package contains a non-ASCII path;
- counts differ from 89 product references or 632 shelf images;
- a copied image differs byte-for-byte from its source;
- the existing labeling script or requirements file differs from the repository source copied into the package.

The existing inference program continues to record individual image failures and returns a nonzero status when failures occur. Its completion-marker and provenance checks continue to govern restart and overwrite behavior within the lifetime of the runtime.

## Verification

Verification will cover:

- the existing offline test suite;
- exact manifest row count, class IDs, barcode values, Thai names, and enabled flags;
- exact one-to-one mapping between 89 manifest rows and 89 ASCII-named references;
- explicit verification of the `/` versus `-` source-name exception;
- 632 decodable shelf images with unique stems;
- a scan proving every path inside the archive is ASCII-only;
- SHA-256 equality between every source image and its prepared copy;
- successful validation-only execution from an extracted copy of the archive;
- valid notebook JSON and a scan proving it contains no Drive or Cloud Storage integration;
- archive extraction and results-packaging checks;
- an optional GPU/model smoke test in Colab Enterprise, because the local verification environment does not guarantee the required checkpoint or CUDA device.

## Deliverables

- Colab Enterprise notebook and supporting configuration in the repository.
- Materialized 89-row manifest and reference mapping.
- English-safe copies of all product references.
- One `hair_colab_runtime.zip` containing the complete runtime-local input package, including all 632 shelf images.
- Updated usage documentation that describes upload, validation, pilot, full run, result download, and runtime-deletion risk.
