# Ultralytics Platform Box-First Annotation Design

## Objective

Replace direct 89-class YOLOE visual-prompt inference with a two-stage annotation assistant that prioritizes tight product boxes and conservative SKU suggestions. The workflow must run in Colab Enterprise runtime storage, require no Google Drive or Cloud Storage access, preserve all source images, and export a standard dataset ZIP that Ultralytics Platform can import with readable English class names.

The output is annotation assistance, not reviewed ground truth. A human must review every imported image in Ultralytics Platform before training.

## Confirmed Problem

The current pipeline treats each complete reference image as the visual-prompt object because `reference_prompts.yaml` has no explicit bounding boxes. The source references are not consistently isolated packshots:

- the median reference resolution is 155 by 212 pixels;
- 79 of 89 references have at least one dimension below 256 pixels;
- some references include hands, store shelves, price labels, or extensive background;
- some references show promotional twin packs or bundles rather than one shelf-facing unit;
- the shelf images contain many small, visually similar products.

Direct visual prompting therefore mixes object appearance with reference background and package context. The large detections observed in the pilot are consistent with this mismatch. Lowering confidence would retain more weak detections and would not correct their geometry.

## Chosen Architecture

The new pipeline separates localization from classification.

### Stage 1: Generic product localization

Each shelf image is divided into overlapping tiles before inference. Default settings are:

- tile size: 1024 pixels;
- overlap: 20 percent;
- full coverage at right and bottom edges;
- deterministic row-major tile order.

YOLOE runs in text-prompt mode with a small generic vocabulary:

- shampoo bottle;
- conditioner bottle;
- hair treatment pouch;
- boxed hair product;
- hair care multipack.

The generic prompts answer only where a product-shaped object is located. They do not assign a permanent SKU class. Instance masks are converted to tight axis-aligned boxes when masks are available; the model box is used when no valid mask exists.

Tile-local boxes are mapped back into original-image coordinates. Cross-tile duplicates are merged with class-agnostic non-maximum suppression. Geometry filters reject invalid or implausible candidates. Configurable defaults reject:

- non-finite or zero-area boxes;
- boxes smaller than 12 pixels on either side in the original image;
- boxes covering more than 10 percent of the full shelf image;
- aspect ratios outside 0.15 to 4.0.

The filters are deliberately configurable because pouches, boxes, bottles, and multipacks have different shapes. Every rejected candidate is counted by reason in the run summary.

### Stage 2: SKU suggestion

The classifier uses the existing product images without editing or replacing the source files. Derived working crops may be generated in runtime storage.

For each reference image, the generic localizer proposes product-shaped regions. The classifier keeps up to three central, high-confidence regions per SKU as derived reference views. If no usable region is found, it uses the full source reference but marks that SKU reference as low quality in diagnostics.

OpenCLIP compares each shelf candidate with:

- the derived visual views for all 89 SKUs; and
- the corresponding concise English SKU name.

The default embedding checkpoint is OpenCLIP `ViT-B-32` with `laion2b_s34b_b79k` pretrained weights. All embeddings are L2-normalized. Each SKU's visual prototype is the normalized mean of its usable reference-view embeddings, and its text prototype uses the template `a retail hair-care product package of {english_sku_name}`. The combined score is `0.80 * image_similarity + 0.20 * text_similarity`.

The notebook downloads both YOLOE and OpenCLIP weights into runtime storage when internet access is available. If model-host access is blocked, the same upload cell accepts local checkpoint files; no Drive mount or Cloud Storage path is introduced.

The model identifier, weights, score weights, and acceptance thresholds live in configuration and are recorded in provenance. The initial pilot defaults are a minimum combined score of 0.24 and a minimum top-one margin of 0.02. These are review-oriented starting points, not calibrated accuracy claims. The policy is conservative:

- accept a permanent SKU only when the top score exceeds the configured minimum and its margin over the second score exceeds the configured minimum margin;
- otherwise assign the temporary `Needs Review` class;
- retain the top three English SKU suggestions and scores in `review_queue.csv` and per-image JSON metadata.

The temporary class is an annotation workflow control, not a trainable business class. It must have zero annotations and be deleted in Ultralytics Platform before training. Because it is the final class at ID 89, deleting it does not renumber permanent IDs 0 through 88.

### Why this architecture

Localization and fine-grained SKU recognition fail for different reasons. Generic localization can learn bottle, pouch, box, and multipack geometry without relying on the poor SKU reference background. SKU matching then operates on tight candidate crops instead of whole shelf regions. Conservative rejection avoids presenting uncertain predictions as trustworthy labels.

## Alternatives Considered

### Continue direct 89-class YOLOE inference with tiling

This is the smallest code change, but it preserves the central failure: each class prompt still contains inconsistent background and pack context. Tiling can make shelf products larger but cannot make a cluttered reference represent a clean object. This option is not selected.

### Use Ultralytics Platform manual and SAM annotation only

This provides strong human control and avoids automatic class errors, but users must create most boxes manually. It remains the fallback for images where the generic localizer performs poorly, not the default workflow.

## Ultralytics Platform Dataset Export

The result packager creates a strict Platform import archive, `hair_ultralytics_platform.zip`, with one dataset root and one `data.yaml`:

```text
hair_ultralytics_platform.zip
├── data.yaml
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
```

Diagnostic files are kept outside the import archive in `hair_annotation_review.zip` so the Platform upload contains only the documented YOLO dataset structure. The review archive contains `review_queue.csv`, per-image JSON metadata, previews, run summaries, and provenance.

Images are assigned to train or validation deterministically from their filename so reruns preserve the split. The default split is 90 percent train and 10 percent validation. Original image bytes and filenames are preserved.

`data.yaml` contains:

- relative `train` and `val` image paths;
- English names for permanent class IDs 0 through 88;
- class ID 89 named `Needs Review`.

YOLO label rows retain the required numeric form:

```text
class_id center_x center_y width height
```

Ultralytics Platform reads the English names from `data.yaml`, so the class sidebar and annotation overlays display product names instead of placeholder identifiers. The original Thai name and barcode remain available in the generated manifest and review metadata.

## Colab Enterprise Workflow

The notebook keeps pilot and full runs separate.

1. Upload and extract the runtime package.
2. Install pinned dependencies and run offline tests.
3. Validate 89 SKU mappings and 632 shelf images.
4. Run generic localization on 10 pilot images.
5. Run SKU suggestion on the pilot candidates.
6. Display pilot previews with English names and confidence scores.
7. Display localization and classification diagnostics.
8. Continue to the full run only after human approval of the pilot.
9. Package the Platform dataset ZIP and separate review ZIP.
10. Download both ZIP files before deleting the runtime.

Pilot output and full-run output use separate directories. Reruns use provenance and completion markers so compatible results can resume, while incompatible configuration or model changes require explicit overwrite.

## Review Experience

In Ultralytics Platform, the user creates a detection dataset and uploads `hair_ultralytics_platform.zip`. The Platform imports the numeric labels and uses `data.yaml` to display English product names.

Review order is:

1. filter or search for `Needs Review` and reassign every such box;
2. correct loose, merged, or partial boxes;
3. verify high-confidence permanent SKU suggestions;
4. add missed products;
5. confirm the `Needs Review` class has zero annotations;
6. delete the temporary class before training.

`review_queue.csv` lists image filename, box coordinates, assigned class, top three English suggestions, scores, geometry diagnostics, and the reason a candidate was sent for review.

## Configuration

The new configuration section exposes only controls that materially affect the workflow:

- tiling size and overlap;
- generic prompt vocabulary;
- generic detection confidence and IoU threshold;
- minimum and maximum candidate dimensions;
- maximum full-image area ratio;
- permitted aspect-ratio range;
- cross-tile NMS threshold;
- embedding model and pretrained-weight identifiers;
- visual and text score weights;
- minimum SKU score;
- minimum top-one margin;
- maximum derived reference views;
- train/validation split percentage and deterministic seed.

All effective settings, dependency versions, model identifiers, and model-weight fingerprints when available are stored in provenance.

## Error Handling and Safety

- Source product images, shelf images, and the Excel workbook are read-only inputs.
- Generated reference crops exist only under runtime output directories.
- A missing or duplicate barcode-to-English mapping is fatal.
- A dataset export is atomic and refuses to replace an existing archive unless overwrite is explicit.
- A shelf image inference failure is recorded without silently creating an empty successful label.
- An image with no candidates receives an explicit reviewed-needed status in the summary.
- Non-finite coordinates, invalid normalized labels, and paths outside the raw output area are rejected.
- The final packager verifies matching image and label stems, numeric class ranges, normalized coordinates, split disjointness, and archive integrity.

## Testing and Acceptance

Unit tests cover:

- deterministic overlapping tile generation and edge coverage;
- conversion between tile-local and original-image coordinates;
- mask-to-box conversion;
- geometry filtering with boundary cases;
- class-agnostic cross-tile NMS;
- embedding score combination, top-three ordering, and uncertainty gating;
- deterministic train/validation splitting;
- `data.yaml` class names and temporary class placement;
- Platform ZIP structure, label normalization, and atomic replacement behavior;
- preservation of the permanent class ID, barcode, English name, and Thai name mapping.

Offline tests use fake model outputs. GPU integration remains a Colab pilot because the local development environment does not contain the model weights or accelerator.

The 10-image pilot must report, rather than assume, the following review measures:

- candidate count per image;
- boxes rejected by each geometry rule;
- proportion assigned to `Needs Review`;
- top-one and top-two score distributions;
- duplicate boxes removed by NMS;
- images with zero candidates;
- preview images in original-image coordinates.

The user decides whether pilot geometry is useful enough to run all 632 images. The design does not claim that zero-shot SKU suggestions are reviewed labels or that the existing references can support high automatic SKU accuracy.

## Deliverables

- updated runtime-local Colab notebook;
- two-stage annotation-assist implementation and tests;
- configuration for tiling, localization, matching, and filtering;
- English `data.yaml` class map for Ultralytics Platform;
- `hair_ultralytics_platform.zip` result packager;
- separate `hair_annotation_review.zip` diagnostics packager;
- review queue and diagnostic summaries;
- updated runtime ZIP and usage documentation.

## Primary References

- [Ultralytics Platform dataset upload format](https://docs.ultralytics.com/platform/data/datasets)
- [Ultralytics Platform annotation editor](https://docs.ultralytics.com/platform/data/annotation)
- [Ultralytics YOLOE visual and text prompts](https://docs.ultralytics.com/models/yoloe)
- [OpenCLIP model loading and pretrained checkpoint identifiers](https://github.com/mlfoundations/open_clip)
