# State-Dependent Lightning Nowcasting

## Study and authors

**State-Dependent Operating Points for Local-Onset and Ongoing Lightning Nowcasting**

Authors: **Wangquan He, Liqun Liu, Qingfeng Wu**.

Affiliation: **College of Electronic Information Engineering, Taiyuan University of Science and Technology, Taiyuan, Shanxi, China**.

This repository contains code and manuscript-specific reproducibility materials associated with the above study. The repository provides the code and frozen manuscript-specific definitions used in the study. The complete end-to-end workflow has not been independently reproduced from a clean environment.

Repository: [state-dependent-lightning-nowcasting](https://github.com/2978264217hd-web/state-dependent-lightning-nowcasting). Version metadata: **1.0.0**. The frozen v1.0.0 software archive is available on Zenodo at **https://doi.org/10.5281/zenodo.23320002**. No paper DOI is assigned yet. No author email or unconfirmed ORCID is published here. Code license: **MIT**; see LICENSE.

## Data access and scope

**Raw GOES GLM and MRMS observations are not redistributed in this repository. Users must obtain the original observations from official NOAA sources.**

GLM: [NOAA NCEI catalog](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C01527). MRMS: [NOAA NSSL](https://www.nssl.noaa.gov/projects/mrms/) and the [NOAA MRMS archive listing](https://registry.opendata.aws/noaa-mrms-pds/). Source selection uses the frozen GOES-16 2024 and GOES-19 2025 inventories, required buffered support and original mask policies. Raw Composite is a QC audit reference rather than a fourth model field.

Only the code package is published in this repository. The separate processed DATA package (frozen normalization/decision parameters, date/split/sample manifests, schema/mask definitions, verification summaries and figure/table numeric inputs) and large per-pixel partitions are **not included here**; their persistent public archive/access route remains pending. Scientific constants and definitions in the supplied implementation are retained. Full inference needs all original/reconstructed confirmatory feature, label and sample-manifest partitions and their frozen hashes. Code availability must not be interpreted as a complete data deposit or full reproduction certification.

## Repository structure

| Path | Actual content |
| --- | --- |
| preprocessing/ | Original semantic/candidate/QC rules, GLM/GRIB decoding, frozen R82 inventories, final content-QC transport, precision recovery, ROI and temporal-support implementations |
| feature_construction/ | Original labels/masks, R55 date splitting and R82 feature/sample/event construction |
| model/ | Original R6 model implementation; 12 small existing frozen checkpoints under checkpoints_r6/ and checkpoints_r7/ |
| decision/ | Original R7 decision implementation and scientific constants; effective frozen parameter files belong to the separate DATA package |
| verification/ | Original frozen R83 inference and verification implementation |
| figures/ | Original figure-generation functions plus the 18 frozen Figure 1–6 PDF/SVG/PNG source assets |
| release_tools.py | Directory/checksum invocation adapter; no new scientific algorithm |
| FUNCTION_PRESERVATION.json | Source-relative function/class provenance and exact-body hashes; this is public reproducibility metadata |
| workspace_map.json | Code/checkpoint mapping into the original relative reproduction layout |
| SHA256SUMS.json | Public snapshot file checksums, excluding this checksum file itself |
| requirements.txt | Major dependencies and evidence-qualified versions |
| CITATION.cff | Confirmed authors/title/repository, version 1.0.0, Zenodo DOI, release date, and MIT license; no unconfirmed ORCID |

Model checkpoint files are retained scientific artifacts, not training caches. No raw observation arrays, training caches, private reports, manuscript/cover letter, credential files, system configuration or administrative audit logs are included. The code's scientific functions/classes are preserved; path configuration is documented below.

## Python environment

Reference research runtime: **CPython 3.13 on Windows**. Frozen R83 records identify **PyTorch 2.7.1+cu118**, **PyArrow 25.0.1** and **DuckDB 1.5.6**. Other major libraries are NumPy, SciPy, netCDF4, cftime, ecCodes and Matplotlib. The complete original pip lock and exact Python microversion are not available. The other dependency versions are therefore not asserted as independently validated pins.

For a future reproduction environment, install the frozen CUDA wheel separately and the listed dependencies:

```shell
python -m pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

ecCodes/NetCDF native support, compatible CUDA drivers, memory and disk capacity are required. Retained code includes an 80 GiB free-space guard and additional external-sort space checks. No environment installation or scientific command was executed as part of this publication action. The software release remains without clean-environment reproduction certification.

## Configurable paths and staging

Choose a **new reproduction workspace outside the repository** and supply a separately obtained processed DATA directory. `--data-root` sets `LIGHTNING_DATA_ROOT`; `--workspace` sets `LIGHTNING_WORKSPACE` for retained code. No private absolute working path is embedded.

```shell
python release_tools.py prepare --workspace reproduction_workspace
python release_tools.py prepare --workspace reproduction_workspace --data-root processed_DATA
```

The first command stages code/checkpoints only. With DATA supplied, the second also stages source-relative metadata and policies. The adapter refuses to replace different existing inputs. It does not restore old R83 output seals automatically. Expected workspace paths include scripts/, config/, checkpoints_r6/, TRAIN_NORMALIZATION_R6.json, R7_METHOD_FREEZE.json, and R8_EXTERNAL_CONFIRMATION/R82_CONFIRMATORY_DATA_BUILD/. Future R83 outputs belong under the corresponding R83_FROZEN_INFERENCE/ directory.

## Preprocessing reproduction route

Use official NOAA source objects, frozen date assignments, source timestamp/age and support rules. Download raw observations separately; do not fill unavailable observations or change policies. The preprocessing scripts preserve the original GRIB semantic classes, float64 GLM decoding and source-time endpoints. Stage names supported by the adapter are `mrms-inventory`, `mrms-content-qc`, `mrms-roi-cache`, `glm-precision`, `raw-composite-audit`, `temporal-support`, `label-readiness`, `events-and-sample-manifest`, and `features`.

Future invocation pattern:

```shell
python release_tools.py stage --workspace reproduction_workspace --data-root processed_DATA --stage STAGE
```

Check each original script's prerequisites first. GLM schema preflight additionally requires the appropriate original R81 NOAA sample granule, downloaded separately with its retained filename. The separate DATA package supplies inventories and freeze/audit records; this code-only repository cannot substitute for them.

Historical preprocessing/finalization entry points contain original whole-file audit hash guards. Relocated paths or sanitized metadata can change container hashes while functions remain unchanged. Those guards may stop a portable rerun; they are not bypassed here. The route therefore requires future engineering validation of audit bookkeeping against release provenance before claiming a complete raw-to-feature reproduction. Do not run training, calibration-selection, threshold-search, final-gate or control-log append entry points to reproduce a submitted frozen result.

## Frozen inference and verification

Once all exact processed inputs and effective parameter files are separately restored, future commands are:

```shell
python release_tools.py inference --workspace reproduction_workspace --data-root processed_DATA
python release_tools.py verification --workspace reproduction_workspace --data-root processed_DATA
```

The adapter calls original frozen load_models/dayrun and evaluation/bootstrap functions. Existing checkpoint tensors, normalization/method and partition hashes are checked; no fitting or threshold search is introduced. Original primary-state routing remains in the QF sensitivity evaluation. Historical domain-shift/control-log bookkeeping is represented by its separately held summary records, not rerun by the adapter. No inference, evaluation, bootstrap or calibration was run to publish this snapshot.

## Manuscript tables and figures

The separate processed DATA package provides exact displayed main/SI table exports and original full-precision figure inputs. Once supplied, future commands are:

```shell
python release_tools.py tables --data-root processed_DATA --output reproduced_tables
python release_tools.py figures --data-root processed_DATA --output reproduced_figures
```

Table export copies existing cell text without computing new metrics. The original plotting program reads the same frozen inputs; output is under reproduced_figures/R10_V2_FIGURES/. The byte-preserved assets already in figures/source_files/ are the authoritative frozen figures. Rendering under a different backend/font/runtime is not certified bit-identical. No figure was regenerated during publication.

## Citation and publication status

For the frozen software used with this study, cite the archived v1.0.0 release at **https://doi.org/10.5281/zenodo.23320002**. The Zenodo concept DOI for all software versions is **https://doi.org/10.5281/zenodo.23320001**. The GitHub repository remains the development and browsing location. No complete end-to-end reproducibility claim is made.
