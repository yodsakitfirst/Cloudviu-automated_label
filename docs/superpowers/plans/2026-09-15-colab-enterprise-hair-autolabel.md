# Colab Enterprise Hair Auto-Label Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runtime-local Colab Enterprise package around the existing YOLOE labeler, with 89 ASCII-named product references, 632 shelf images, and one downloadable upload archive.

**Architecture:** Leave `yoloe_autolabel.py` unchanged as the inference engine. Add a tested packaging utility that reads `Sheet2`, validates and maps the supplied data, copies product references to deterministic `class_<id>_<barcode>` names, and creates one archive; add a thin notebook that extracts the archive and invokes the existing CLI for tests, validation, pilot, full inference, and result packaging.

**Tech Stack:** Python 3.11–3.12, standard library, openpyxl, PyYAML, OpenCV, pytest, Jupyter notebook JSON, existing Ultralytics YOLOE pipeline.

**Spec:** `docs/superpowers/specs/2026-09-15-colab-enterprise-hair-autolabel-design.md`

## Global Constraints

- Use exactly the 89 `Sheet2` rows, in source row order, as permanent class IDs `0` through `88`.
- Run all 89 enabled classes against every one of the 632 shelf images; supermarket columns never filter inference.
- Preserve barcode values as text and preserve Thai product names as data.
- Use ASCII-only paths inside the prepared archive.
- Do not rename or modify anything under `/Users/me/Downloads/Cloudviu-data`.
- Do not mount Google Drive or use Cloud Storage, `gs://`, GCS clients, or bucket synchronization.
- Do not edit `yoloe_autolabel.py`; verify that the archive copy is byte-identical to the repository file.
- Keep pilot and full-run output roots separate.
- Keep generated annotations under `raw_predictions`; never write or package them as reviewed ground truth.

---

### Task 1: Parse and Match the Hair Product Sources

**Files:**
- Create: `prepare_hair_colab.py`
- Create: `tests/test_prepare_hair_colab.py`
- Create: `colab/reference_name_overrides.yaml`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `HairSku(class_id: int, barcode: str, sku_name: str)`, `normalize_product_name(value: str) -> str`, `load_sheet2_skus(path: Path, expected_count: int = 89) -> list[HairSku]`, `load_reference_overrides(path: Path) -> dict[str, str]`, `match_product_references(skus, product_dir, overrides) -> dict[int, Path]`, and `ascii_reference_name(sku, source) -> str`.
- Consumes: workbook `Sheet2` rows 3–91 and the supplied product-image directory.

- [ ] **Step 1: Add the workbook and normalization tests**

Create test fixtures with `openpyxl.Workbook`, putting the actual headers in row 2, and add these tests:

```python
def write_workbook(path: Path, rows: list[tuple[str, str]]) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sheet2"
    sheet.append([None, None])
    sheet.append(["Barcode", "Product Name"])
    for barcode, name in rows:
        sheet.append([barcode, name])
    workbook.save(path)


def write_image(path: Path) -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_load_sheet2_skus_preserves_row_order_barcodes_and_thai_names(tmp_path):
    workbook = tmp_path / "sku.xlsx"
    write_workbook(workbook, [
        ("08851932487177", "สินค้า หนึ่ง"),
        ("18851932355343", "สินค้า สอง"),
    ])

    skus = prep.load_sheet2_skus(workbook, expected_count=2)

    assert skus == [
        prep.HairSku(0, "08851932487177", "สินค้า หนึ่ง"),
        prep.HairSku(1, "18851932355343", "สินค้า สอง"),
    ]


def test_load_sheet2_skus_rejects_missing_duplicate_and_non_digit_identifiers(tmp_path):
    cases = [
        [("8851932487177", "A"), ("8851932487177", "B")],
        [("8851932487177", "A"), ("", "B")],
        [("885193248717X", "A")],
        [("123", "A")],
        [("8851932487177", "A"), ("8851932487178", "A")],
    ]
    for index, rows in enumerate(cases):
        workbook = tmp_path / f"invalid-{index}.xlsx"
        write_workbook(workbook, rows)
        with pytest.raises(ValueError):
            prep.load_sheet2_skus(workbook, expected_count=len(rows))


def test_normalize_product_name_is_unicode_and_whitespace_stable():
    assert prep.normalize_product_name("  สิ\u0e19ค้า\tหนึ่ง  ") == prep.normalize_product_name("สินค้า หนึ่ง")
```

- [ ] **Step 2: Run the new tests and confirm the expected failure**

Run: `python -m pytest tests/test_prepare_hair_colab.py -q`

Expected: collection fails because `prepare_hair_colab` does not exist.

- [ ] **Step 3: Implement the minimal workbook parser and identifier validation**

Add the following public shape to `prepare_hair_colab.py`, using `openpyxl.load_workbook(read_only=True, data_only=True)` and reading columns A and B from row 3 until both are blank:

```python
@dataclass(frozen=True)
class HairSku:
    class_id: int
    barcode: str
    sku_name: str


def normalize_product_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split()).casefold()


def _barcode_text(value: object, row_number: int) -> str:
    if isinstance(value, bool):
        raise ValueError(f"Invalid barcode in Sheet2 row {row_number}")
    if isinstance(value, int):
        barcode = str(value)
    elif isinstance(value, float) and value.is_integer():
        barcode = str(int(value))
    else:
        barcode = str(value or "").strip()
    if not barcode.isascii() or not barcode.isdigit() or len(barcode) not in {13, 14}:
        raise ValueError(f"Invalid barcode in Sheet2 row {row_number}: {barcode!r}")
    return barcode
```

`load_sheet2_skus` must reject missing `Sheet2`, incorrect row count, blank values, duplicate barcodes, and duplicate normalized names.

- [ ] **Step 4: Run the parser tests and confirm they pass**

Run: `python -m pytest tests/test_prepare_hair_colab.py -q`

Expected: all parser tests pass.

- [ ] **Step 5: Add failing reference-matching tests**

```python
def test_match_product_references_uses_exact_normalized_names_and_explicit_override(tmp_path):
    products = tmp_path / "products"
    products.mkdir()
    exact = products / "สินค้า หนึ่ง.png"
    override = products / "สินค้า-สอง.jpg"
    exact.write_bytes(b"exact")
    override.write_bytes(b"override")
    skus = [prep.HairSku(0, "8851932487177", "สินค้า หนึ่ง"), prep.HairSku(1, "8851932487178", "สินค้า/สอง")]

    matched = prep.match_product_references(
        skus,
        products,
        {"สินค้า/สอง": "สินค้า-สอง.jpg"},
    )

    assert matched == {0: exact.resolve(), 1: override.resolve()}
    assert prep.ascii_reference_name(skus[0], exact) == "class_000_8851932487177.png"


def test_match_product_references_rejects_missing_ambiguous_and_unused_overrides(tmp_path):
    products = tmp_path / "products"
    products.mkdir()
    (products / "A.jpg").write_bytes(b"a")
    (products / "a.png").write_bytes(b"b")
    with pytest.raises(ValueError, match="Ambiguous"):
        prep.match_product_references([prep.HairSku(0, "8851932487177", "A")], products, {})
    with pytest.raises(ValueError, match="Unused override"):
        prep.match_product_references([prep.HairSku(0, "8851932487177", "B")], products, {"X": "A.jpg"})
```

- [ ] **Step 6: Run the matching tests and confirm they fail for missing behavior**

Run: `python -m pytest tests/test_prepare_hair_colab.py -q`

Expected: failures report missing matching and naming functions.

- [ ] **Step 7: Implement strict matching and the supplied explicit override**

Support only `.jpg`, `.jpeg`, and `.png`; ignore `.DS_Store`; reject duplicate normalized stems, missing matches, multiple images per SKU, an override target outside the product directory, and any unused override. Create `colab/reference_name_overrides.yaml` with:

```yaml
overrides:
  "เคลียร์แชมพูผช คูล 25บ/29B 12X6X60มล.": "เคลียร์แชมพูผช คูล 25บ-29B 12X6X60มล..png"
```

Add `openpyxl>=3.1,<4` to `requirements.txt` so the preparation command is reproducible.

- [ ] **Step 8: Run the focused and existing tests**

Run: `python -m pytest tests/test_prepare_hair_colab.py tests/test_yoloe_autolabel.py -q`

Expected: all tests pass.

- [ ] **Step 9: Commit the parsing and matching layer**

```bash
git add prepare_hair_colab.py tests/test_prepare_hair_colab.py colab/reference_name_overrides.yaml requirements.txt
git commit -m "feat: validate hair SKU sources"
```

---

### Task 2: Build a Safe, Reproducible Runtime Archive

**Files:**
- Modify: `prepare_hair_colab.py`
- Modify: `tests/test_prepare_hair_colab.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `HairSku` rows and matched source paths from Task 1; `repo_root/yoloe_autolabel.py`, `repo_root/requirements.txt`, `repo_root/tests/test_yoloe_autolabel.py`, `repo_root/colab/config.yaml`, and `repo_root/colab/hair_colab_enterprise.ipynb`.
- Produces: `PackageInputs(workbook: Path, product_images: Path, shelf_images: Path, repo_root: Path, overrides: Path, expected_skus: int = 89, expected_shelves: int = 632)`, `PackageReport(archive_path: Path, sku_count: int, reference_count: int, shelf_count: int)`, `discover_shelf_images(path: Path, expected_count: int = 632) -> list[Path]`, `build_runtime_package(inputs: PackageInputs, destination: Path, overwrite: bool = False) -> PackageReport`, and CLI flags `--workbook`, `--product-images`, `--shelf-images`, `--repo-root`, `--output`, `--overrides`, `--expected-skus`, `--expected-shelves`, and `--overwrite`.

- [ ] **Step 1: Add failing shelf-discovery and archive-safety tests**

```python
def test_discover_shelf_images_selects_only_unique_decodable_jpg_files(tmp_path):
    shelves = tmp_path / "shelves"
    shelves.mkdir()
    write_image(shelves / "a.jpg")
    write_image(shelves / "b.JPG")
    (shelves / "index.csv").write_text("ignored", encoding="utf-8")
    assert [p.name for p in prep.discover_shelf_images(shelves, expected_count=2)] == ["a.jpg", "b.JPG"]


def test_build_runtime_package_copies_bytes_and_uses_ascii_archive_paths(package_fixture, tmp_path):
    destination = tmp_path / "hair_colab_runtime.zip"
    report = prep.build_runtime_package(package_fixture, destination)

    assert report.sku_count == 2
    assert report.reference_count == 2
    assert report.shelf_count == 2
    with zipfile.ZipFile(destination) as archive:
        assert all(info.filename.isascii() for info in archive.infolist())
        assert archive.read("hair_colab/yoloe_autolabel.py") == package_fixture.engine.read_bytes()
        assert archive.read("hair_colab/references/class_000_8851932487177.png") == package_fixture.references[0].read_bytes()
```

Build `package_fixture` from two generated workbook rows, two valid product images, two valid shelf images, and a fake repository tree containing the five exact required files listed in **Interfaces**. Also test that an existing destination fails without `overwrite`, an invalid/corrupt shelf image fails, duplicate case-insensitive shelf stems fail, non-ASCII shelf filenames fail, and failure leaves no final ZIP.

- [ ] **Step 2: Run the archive tests and confirm the expected failures**

Run: `python -m pytest tests/test_prepare_hair_colab.py -q`

Expected: failures identify missing `PackageInputs`, discovery, and build interfaces.

- [ ] **Step 3: Implement atomic package construction**

Use a temporary directory beside the destination, build `hair_colab/` there, verify it, create a temporary ZIP, then use `os.replace` for the final archive. Copy files with `shutil.copy2`; compare source and destination SHA-256 digests after every image copy. Generate the manifest with `csv.DictWriter` and this exact header:

```python
MANIFEST_FIELDS = ("class_id", "barcode", "brand", "sku_name", "enabled")
```

Write `brand="Unspecified"` and `enabled="true"`. Generate reference YAML in ascending class-ID order with relative paths such as `references/class_000_8851932487177.png`. Copy only the 632 discovered shelf `.jpg` files.

Before ZIP creation, walk the stage tree and enforce:

```python
for path in stage_root.rglob("*"):
    relative = path.relative_to(stage_root)
    if not relative.as_posix().isascii():
        raise ValueError(f"Prepared path is not ASCII: {relative}")
```

The `--overwrite` path may replace only the explicit archive destination. It must not delete or modify any source directory.

- [ ] **Step 4: Add the command-line entry point**

The CLI must end with:

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

On success, print the archive path and the three verified counts. On validation or I/O failure, print one `ERROR:` line to stderr and return `2`.

- [ ] **Step 5: Ignore generated package artifacts**

Add these anchored entries to `.gitignore`:

```gitignore
/dist/
/colab/generated/
```

- [ ] **Step 6: Run focused and full offline tests**

Run: `python -m pytest tests/test_prepare_hair_colab.py -q`

Expected: all preparation tests pass.

Run: `python -m pytest -q`

Expected: all repository tests pass.

- [ ] **Step 7: Commit the archive builder**

```bash
git add prepare_hair_colab.py tests/test_prepare_hair_colab.py .gitignore
git commit -m "feat: build Colab runtime archive"
```

---

### Task 3: Add Runtime-Relative Configuration and Notebook Orchestration

**Files:**
- Create: `colab/config.yaml`
- Create: `colab/hair_colab_enterprise.ipynb`
- Create: `tests/test_colab_assets.py`
- Modify: `prepare_hair_colab.py`

**Interfaces:**
- Consumes: an uploaded `hair_colab_runtime.zip` and the existing `yoloe_autolabel.py` CLI.
- Produces: notebook cells for extraction, environment checks, dependency installation, tests, 89-class validation, pilot, full run, previews, and `hair_label_results.zip` creation.

- [ ] **Step 1: Add failing configuration and notebook contract tests**

```python
def test_colab_config_uses_one_manifest_and_runtime_relative_paths():
    config = yaml.safe_load(Path("colab/config.yaml").read_text(encoding="utf-8"))
    assert config["dataset"] == {
        "sku_manifest": "sku_manifest.csv",
        "references": "reference_prompts.yaml",
        "shelf_images": "shelf_images",
        "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],
    }
    assert config["output"]["root"] == "output"
    assert config["pilot"]["enabled"] is False


def test_colab_notebook_is_runtime_local_and_contains_required_commands():
    notebook = json.loads(Path("colab/hair_colab_enterprise.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "hair_colab_runtime.zip" in source
    assert "--require-enabled-count" in source and "89" in source
    assert "pilot_output" in source
    assert "hair_label_results.zip" in source
    for forbidden in ("drive.mount", "google.colab", "gs://", "google.cloud.storage", "/content"):
        assert forbidden not in source
```

Also assert that one markdown cell warns that runtime files are deleted and that the full-run command uses `config.yaml` without `--overwrite`.

- [ ] **Step 2: Run the asset tests and confirm they fail because the files are absent**

Run: `python -m pytest tests/test_colab_assets.py -q`

Expected: failures report missing `colab/config.yaml` and notebook.

- [ ] **Step 3: Create the full-run configuration**

Copy the existing model settings unless validation proves them incompatible, set `project.name: hair_osa_89`, use the relative dataset paths asserted above, keep all 89 manifest rows enabled, and set:

```yaml
pilot:
  enabled: false
  class_ids: []
  max_skus: 5
  max_images: 10
```

- [ ] **Step 4: Create the notebook with explicit, separately executable cells**

The extraction cell must locate the uploaded archive without a platform-specific root:

```python
from pathlib import Path
import shutil

archives = sorted(Path.cwd().rglob("hair_colab_runtime.zip"))
if len(archives) != 1:
    raise RuntimeError(f"Expected one uploaded hair_colab_runtime.zip, found {len(archives)}")
workspace = Path.cwd() / "hair_colab_workspace"
if workspace.exists():
    raise RuntimeError(f"Workspace already exists: {workspace}")
shutil.unpack_archive(archives[0], workspace)
project = workspace / "hair_colab"
```

Use `subprocess.run([...], cwd=project, check=True)` rather than shell magics for install, tests, validation, pilot, and full run. Build `pilot_config.yaml` by loading `config.yaml`, setting `pilot.enabled=true`, `pilot.max_skus=5`, `pilot.max_images=10`, and `output.root="pilot_output"`. The full-run cell must call:

```python
subprocess.run(
    [sys.executable, "yoloe_autolabel.py", "--config", "config.yaml", "--require-enabled-count", "89"],
    cwd=project,
    check=True,
)
```

The results cell must create `hair_label_results.zip` beside the extracted workspace with `shutil.make_archive`, then print its path and a reminder to download it from the Files pane before deleting the runtime.

- [ ] **Step 5: Teach the package builder to copy the Colab assets**

Add `colab/config.yaml` and `colab/hair_colab_enterprise.ipynb` to the required repository inputs copied into the archive. Include only `tests/test_yoloe_autolabel.py` in the runtime test directory; preparation tests stay in the development repository because the source workbook is intentionally excluded from the archive.

- [ ] **Step 6: Run notebook, builder, and complete test suites**

Run: `python -m pytest tests/test_colab_assets.py tests/test_prepare_hair_colab.py -q`

Expected: all Colab and package tests pass.

Run: `python -m pytest -q`

Expected: all repository tests pass.

- [ ] **Step 7: Commit the Colab runtime assets**

```bash
git add colab/config.yaml colab/hair_colab_enterprise.ipynb prepare_hair_colab.py tests/test_colab_assets.py
git commit -m "feat: add Colab Enterprise workflow"
```

---

### Task 4: Materialize and Verify the Supplied 89-Class Dataset

**Files:**
- Generated, ignored: `colab/generated/sku_manifest.csv`
- Generated, ignored: `colab/generated/reference_prompts.yaml`
- Generated, ignored: `dist/hair_colab_runtime.zip`
- Generated, temporary: extracted verification directory under `/tmp`

**Interfaces:**
- Consumes: the three supplied paths and the preparation CLI.
- Produces: the exact user-ready archive and inspectable generated registry files through CLI option `--metadata-output PATH`.

- [ ] **Step 1: Add a failing real-data report option test**

Add a test proving `--metadata-output` writes only the generated manifest and reference YAML after a successful build and does not copy source images there:

```python
def test_cli_metadata_output_contains_only_registry_files(package_fixture, tmp_path):
    metadata = tmp_path / "generated"
    destination = tmp_path / "hair_colab_runtime.zip"
    result = prep.main([
        "--workbook", str(package_fixture.workbook),
        "--product-images", str(package_fixture.product_images),
        "--shelf-images", str(package_fixture.shelf_images),
        "--repo-root", str(package_fixture.repo_root),
        "--overrides", str(package_fixture.overrides),
        "--output", str(destination),
        "--metadata-output", str(metadata),
        "--expected-skus", "2",
        "--expected-shelves", "2",
    ])
    assert result == 0
    assert sorted(p.name for p in metadata.iterdir()) == ["reference_prompts.yaml", "sku_manifest.csv"]
```

- [ ] **Step 2: Run the focused test and confirm the missing-option failure**

Run: `python -m pytest tests/test_prepare_hair_colab.py::test_cli_metadata_output_contains_only_registry_files -q`

Expected: failure reports that `--metadata-output` is unsupported.

- [ ] **Step 3: Implement atomic metadata export**

Write both registry files to a temporary sibling directory and rename it into place only after the archive succeeds. Refuse to replace an existing metadata directory unless `--overwrite` is present. Reuse the exact manifest and reference-YAML bytes written into the archive.

- [ ] **Step 4: Run the package builder on the supplied sources**

Run:

```bash
python prepare_hair_colab.py \
  --workbook "/Users/me/Downloads/Cloudviu-data/HAIR SKU list (with Account list).xlsx" \
  --product-images "/Users/me/Downloads/Cloudviu-data/HAIR_product_images" \
  --shelf-images "/Users/me/Downloads/Cloudviu-data/HAIR_shelf_images" \
  --repo-root "/Users/me/Cloudviu-automated_label" \
  --overrides "/Users/me/Cloudviu-automated_label/colab/reference_name_overrides.yaml" \
  --metadata-output "/Users/me/Cloudviu-automated_label/colab/generated" \
  --output "/Users/me/Cloudviu-automated_label/dist/hair_colab_runtime.zip"
```

Expected: success reports 89 SKUs, 89 references, and 632 shelf images.

- [ ] **Step 5: Independently inspect the materialized registry**

Run a read-only verification command that asserts:

```python
rows = list(csv.DictReader(open("colab/generated/sku_manifest.csv", encoding="utf-8")))
assert len(rows) == 89
assert [int(row["class_id"]) for row in rows] == list(range(89))
assert all(row["enabled"] == "true" for row in rows)
assert all(row["barcode"].isdigit() and len(row["barcode"]) in {13, 14} for row in rows)
```

Inspect the exceptional Clear SKU row and confirm it points to its `class_<id>_<barcode>.png` reference.

- [ ] **Step 6: Extract and validate the actual archive**

Extract to a new `mktemp -d` directory. From `hair_colab/`, run:

```bash
python yoloe_autolabel.py --config config.yaml --validate-only --require-enabled-count 89
```

Expected: `Validation successful: 89 SKU(s), 632 image(s)`.

- [ ] **Step 7: Verify the actual archive paths and bytes**

Use a Python read-only check over `zipfile.ZipFile.infolist()` to assert every member name is ASCII, there are exactly 89 `references/` image members and 632 `shelf_images/` members, the engine bytes equal repository `yoloe_autolabel.py`, and every archived image hash equals its corresponding source hash.

- [ ] **Step 8: Run the complete test suite after materialization**

Run: `python -m pytest -q`

Expected: all tests pass with no warnings or errors.

---

### Task 5: Replace the Placeholder Documentation with the Actual Colab Workflow

**Files:**
- Modify: `README.md`
- Create: `colab/README.md`

**Interfaces:**
- Consumes: the verified archive, notebook, config, and existing script behavior.
- Produces: exact user steps for importing the notebook, uploading the archive, validating, piloting, running, resuming, and downloading results.

- [ ] **Step 1: Update the repository status and dataset facts**

Remove the obsolete statements that the workbook and images are missing and that the run is blocked at 79 classes. State the verified 89-class/632-image scope, the permanent `Sheet2` row-order rule, the English-safe reference naming rule, and that all classes apply to all supermarkets.

- [ ] **Step 2: Add concise Colab Enterprise instructions**

Document this exact sequence:

1. Import `colab/hair_colab_enterprise.ipynb` into Colab Enterprise.
2. Connect to a GPU runtime with sufficient disk.
3. Upload `dist/hair_colab_runtime.zip` through the Files pane.
4. Execute the notebook cells through environment checks, installation, tests, and validation.
5. Run and inspect the pilot.
6. Execute the separately gated full-run cell.
7. Create and download `hair_label_results.zip` before runtime deletion.

State that the official checkpoint alias needs public internet on first use; with restricted internet, upload the checkpoint and edit only `yoloe.model` in `config.yaml` to its package-relative path.

- [ ] **Step 3: Document resume and review boundaries**

Explain that rerunning without overwrite skips completed label markers only while compatible provenance and runtime files remain. Explain that runtime deletion removes uploaded and generated files, and that raw labels require human correction before training.

- [ ] **Step 4: Run documentation contract scans**

Run:

```bash
rg -n "89|632|hair_colab_runtime.zip|hair_label_results.zip|runtime" README.md colab/README.md
rg -n "drive.mount|google.colab|gs://|google.cloud.storage|/content" README.md colab/README.md colab/hair_colab_enterprise.ipynb
```

Expected: the first command finds every required workflow term; the second command returns no matches.

- [ ] **Step 5: Commit the documentation**

```bash
git add README.md colab/README.md
git commit -m "docs: explain runtime-local Colab labeling"
```

---

### Task 6: Final Verification and Handoff

**Files:**
- Verify: all tracked files
- Verify: `dist/hair_colab_runtime.zip`
- Verify: `colab/generated/sku_manifest.csv`
- Verify: `colab/generated/reference_prompts.yaml`

**Interfaces:**
- Consumes: every deliverable from Tasks 1–5.
- Produces: verification evidence and a ready-to-upload archive; no GPU accuracy claim.

- [ ] **Step 1: Run the full offline suite**

Run: `python -m pytest -q`

Expected: all tests pass with a zero exit status.

- [ ] **Step 2: Run CLI and syntax checks**

Run:

```bash
python -m py_compile yoloe_autolabel.py prepare_hair_colab.py
python yoloe_autolabel.py --help
python prepare_hair_colab.py --help
```

Expected: compilation and both help commands succeed.

- [ ] **Step 3: Rebuild from a clean generated-output state**

Move the existing ignored `dist/` and `colab/generated/` artifacts to a temporary backup, run the Task 4 package command without `--overwrite`, verify success, and delete only the temporary backup after the rebuilt archive passes all checks.

- [ ] **Step 4: Re-run archive and extracted validation checks**

Repeat Task 4 Steps 6 and 7. Confirm 89 classes, 89 references, 632 shelf images, ASCII-only archive paths, source/copy hash equality, and byte-identical `yoloe_autolabel.py`.

- [ ] **Step 5: Inspect repository scope**

Run:

```bash
git status --short
git diff --check
git diff --stat HEAD~4..HEAD
```

Expected: no unintended source-data changes, no edits to `yoloe_autolabel.py`, and no generated ZIP or copied image dataset staged in Git.

- [ ] **Step 6: Record remaining runtime-only validation**

Report that model download, CUDA inference, visual pilot quality, and full-run duration must be verified in the user's Colab Enterprise GPU runtime. Do not describe offline structural validation as model-accuracy validation.
