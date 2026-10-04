# ResQ-AI: Multimodal Data Audit & Truth Table

This document provides a comprehensive, transparent audit of all eight modalities and data layers utilized in ResQ-AI, detailing whether the Google Colab GPU execution pipeline (`notebooks/colab_full_experiments.ipynb`) consumes **real physical/satellite data** or **synthetic/fallback/zeroed data**, their precise data origins, and proposed standalone companion scripts to ingest real data for modalities currently operating on placeholders.

---

## 1. Modality Truth Table

| Modality | Colab Run Status | Data Source / Origin | Resolution & Format | Status in Current Colab Pipeline | Fallback / Behavior when Missing |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SAR (Sentinel-1)** | **REAL** | Google Cloud Storage: `gs://sen1floods11/v1.1/` (Bonafilia et al., 2020) | 10m, GeoTIFF, 2 bands (VV, VH) | Downloaded directly via `gsutil` in Cell 8 into `data/sen1floods11/v1.1/` | Enforces real data; raises `FileNotFoundError` if files missing |
| **Optical (Sentinel-2)** | **REAL** | Google Cloud Storage: `gs://sen1floods11/v1.1/` (Bonafilia et al., 2020) | 10m (harmonized), GeoTIFF, 13 bands (B1–B12, B8A) | Downloaded directly via `gsutil` in Cell 8 into `data/sen1floods11/v1.1/` | Enforces real data; raises `FileNotFoundError` if files missing |
| **Ground Truth Flood Labels** | **REAL** | Google Cloud Storage: `gs://sen1floods11/v1.1/` (Quality-controlled hand labels) | 10m, GeoTIFF, 1 band (0=no flood, 1=flood, -1=nodata) | Downloaded directly via `gsutil` in Cell 8 into `data/sen1floods11/v1.1/` | Enforces real data; raises `FileNotFoundError` if files missing |
| **DEM / Terrain (Elevation, Slope)** | **PLACEHOLDER / ZEROED** | Copernicus GLO-30 Global DEM (AWS Open Data / ESA) | 30m, COG / GeoTIFF | Not downloaded in Colab Cell 8; `MultiModalFloodDataset` defaults `geo` to zero tensor | Modality mask sets `mask[2] = 0.0`; model trained with modality dropout handles missingness gracefully |
| **HAND & TWI (Hydrological Indices)** | **PLACEHOLDER / ZEROED** | Derived hydrologically from GLO-30 DEM via D8 flow accumulation | 30m derived, 2 channels (HAND, TWI) | Not downloaded/computed for real chips in Colab; zeroed in `geo` feature tensor | Modality mask sets `mask[2] = 0.0`; zero tensor passed to `GeoEncoder` |
| **CHIRPS Rainfall** | **PLACEHOLDER / ZEROED** | UCSB Climate Hazards Center (CHIRPS Daily 0.05°) | 0.05° (~5.3 km), 30-day daily sequence `(30, 1)` | Not downloaded in Colab Cell 8; `rainfall` defaults to zeros `(30, 1)` | Modality mask sets `mask[4] = 0.0`; zero sequence passed to `RainfallEncoder` |
| **ESA WorldCover** | **PLACEHOLDER / ZEROED** | ESA WorldCover 2020/2021 (AWS Open Data) | 10m, GeoTIFF, 11 land cover classes | Not downloaded in Colab Cell 8; zeroed in `geo` feature tensor | Modality mask sets `mask[3] = 0.0`; zero tensor passed to `GeoEncoder` |
| **OSM Road Networks** | **FALLBACK / HAVERSINE** | OpenStreetMap via `osmnx` Overpass API | Vector road graph / NetworkX | Offline Colab environments fail Overpass queries; falls back to Haversine distance matrix | `src/optimization/travel_time.py` lines 60–80 compute great-circle distance at assumed 30 km/h speed |
| **WorldPop Population Exposure** | **FALLBACK / SYNTHETIC** | WorldPop Global 2020 100m unconstrained count | 100m, GeoTIFF | GeoTIFFs not loaded in current Colab experiment scripts; impact uses synthetic spatial exposure | `src/impact/assessment.py` falls back to quadrant-based population density heuristics |

---

## 2. In-Depth Modality Breakdown

### 2.1 SAR (Sentinel-1) — REAL
- **Implementation**: `src/data/sen1floods11.py:Sen1Floods11Dataset`
- **Data Details**: Synthetic Aperture Radar C-band backscatter in Interferometric Wide swath (IW) mode. Dual-polarization VV (vertical transmit/receive) and VH (vertical transmit/horizontal receive). Orthorectified and radiometrically calibrated.
- **Normalization**: Standardized using Sen1Floods11 publication statistics ($\mu_{\text{VV}} = -12.54, \sigma_{\text{VV}} = 5.25, \mu_{\text{VH}} = -20.19, \sigma_{\text{VH}} = 5.73\text{ dB}$).
- **Verification**: Colab Cell 8 executes `gsutil -m cp -r gs://sen1floods11/v1.1/...` fetching 446 hand-labeled chips totaling ~3 GB. Non-smoke mode checks file existence on disk (`self.chips[0]['s1'].exists()`) and throws `FileNotFoundError` if missing.

### 2.2 Optical (Sentinel-2) — REAL
- **Implementation**: `src/data/sen1floods11.py:Sen1Floods11Dataset`
- **Data Details**: Multi-Spectral Instrument (MSI) Level-1C top-of-atmosphere reflectance across 13 spectral channels (coastal aerosol, blue, green, red, red edge 1-3, NIR, narrow NIR, water vapor, SWIR 1-2). Resampled to a common 10m grid.
- **Normalization**: Per-band mean and standard deviation from Sen1Floods11 catalog metadata.
- **Verification**: Downloaded alongside SAR in Colab Cell 8.

### 2.3 DEM & Hydrological Indices (GLO-30, HAND, TWI) — CURRENTLY ZEROED IN COLAB
- **Implementation**: Algorithms exist in `src/data/terrain.py` (`compute_slope`, `compute_hand`, `compute_twi`, `compute_flow_accumulation`, `derive_all_terrain_features`).
- **Reason for Placeholder in Colab**: The Sen1Floods11 Google Cloud Storage bucket (`gs://sen1floods11/v1.1/`) provides only Sentinel-1, Sentinel-2, and Label GeoTIFFs. It does not package matched Copernicus DEM chips. In `src/data/sen1floods11.py:MultiModalFloodDataset`, if `dem_dir` is not provided or files do not exist, `sample["geo"]` is populated with `torch.zeros(6, H, W)` and `modality_mask[2] = 0.0`.
- **Architectural Handling**: `ResQNet` was specifically designed with FiLM conditioning and modality dropout ($p_{\text{drop}} = 0.2$) to handle missing modalities robustly. However, the model does not yet learn physical terrain gradients when trained solely on the GCS download bundle.

### 2.4 CHIRPS Antecedent Rainfall — CURRENTLY ZEROED IN COLAB
- **Implementation**: `src/data/download.py:download_chirps` contains an initial stub.
- **Reason for Placeholder in Colab**: CHIRPS precipitation grids are distributed as daily global NetCDF files (~0.05° resolution) by UCSB. Because downloading and slicing multi-day global NetCDFs takes significant time, Colab Cell 8 does not fetch them, and `sample["rainfall"]` defaults to `torch.zeros(30, 1)` with `modality_mask[4] = 0.0`.
- **Architectural Handling**: The `RainfallEncoder` receives zeros and its FiLM modulation modulates features with baseline scaling.

### 2.5 ESA WorldCover — CURRENTLY ZEROED IN COLAB
- **Implementation**: `src/data/download.py:download_worldcover` contains an initial stub.
- **Reason for Placeholder in Colab**: WorldCover 2020 tiles are 10m global Cloud-Optimized GeoTIFFs (COGs) stored on AWS S3. Colab does not download them, so `sample["modality_mask"][3] = 0.0`.

### 2.6 OpenStreetMap Road Networks — FALLBACK IN COLAB
- **Implementation**: `src/optimization/travel_time.py:compute_travel_times`
- **Reason for Fallback in Colab**: Querying Overpass API on Colab dynamically for urban road graphs often fails or triggers rate limits during automated benchmark execution. When `ox.graph_from_polygon` or bounding box queries fail, lines 60–80 fall back to Haversine great-circle distances between zone centroids and depot coordinates, assuming a nominal response speed of 30 km/h.

### 2.7 WorldPop Population Exposure — FALLBACK IN COLAB
- **Implementation**: `src/impact/assessment.py:sample_flood_maps` and `compute_impact`
- **Reason for Fallback in Colab**: National WorldPop 100m population GeoTIFFs (~1 GB per country) are not bundled in the Sen1Floods11 GCS repository. The Monte Carlo impact module computes demand distributions across spatial quadrants using calibrated flood probabilities multiplied by zonal exposure densities, rather than direct raster zonal statistics against real WorldPop GeoTIFFs.

---

## 3. Proposed Additive Solutions (Standalone Helper Scripts)

To upgrade the remaining placeholder modalities to real physical datasets **without modifying any existing codebase files**, the following new standalone scripts are proposed:

### Proposed Script 1: `scripts/download_copernicus_dem_real.py`
- **Function**: Reads bounding boxes and CRS from each downloaded `*_S1Hand.tif` chip using `rasterio`.
- **Data Access**: Accesses the public Copernicus GLO-30 DEM COGs via AWS Open Data:
  `s3://copernicus-dem-30m/Copernicus_DSM_COG_10_...` (or via OpenTopography API / Planetary Computer STAC).
- **Processing**: Crops DEM to chip extent, runs `src.data.terrain.derive_all_terrain_features()`, and saves pre-computed arrays to `data/dem/{region}_{chip_id}_terrain.npy`.

### Proposed Script 2: `scripts/download_chirps_real.py`
- **Function**: Uses `requests` or `urllib` to query the UCSB CHIRPS daily precipitation NetCDF files for the 30 days prior to each disaster event date:
  `https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/`
- **Processing**: Extracts the rainfall time series for each chip centroid and saves as `data/rainfall/{region}_{chip_id}_rain.npy`.

### Proposed Script 3: `scripts/cache_osm_networks.py`
- **Function**: Queries OSMnx once for the target study regions (e.g. Dartmouth flood bounding boxes, Ernakulam/Kochi, Kerala, Santa Cruz, Bolivia), builds the directed drivable road network, computes shortest-path Dijkstra travel time matrices, and exports them as static `.graphml` or `.parquet` files in `data/osm/`.

### Proposed Script 4: `scripts/download_worldpop_real.py`
- **Function**: Downloads unconstrained 100m individual country population GeoTIFFs from `https://data.worldpop.org/GIS/Population/Global_2020_2021_100m_UNadj/` for target countries (e.g., India, Bolivia, Spain, USA) and extracts population counts per demand zone.

---

## 4. Summary & Scientific Governance

- **Computer Vision Segmentation (SAR + Optical + Hand Labels)**: Operates on **100% REAL** Sentinel-1, Sentinel-2, and quality-controlled ground truth masks in the Colab GPU benchmark.
- **Ancillary Earth Observation (DEM, LandCover, Rainfall)**: Operates with **modality masks disabled / zeroed** in the baseline Sen1Floods11 Colab run due to absence in the official GCS bucket.
- **Downstream Operations (Road Networks, Population Exposure)**: Evaluated using **calibrated Monte Carlo flood distributions** coupled with Haversine travel times and synthetic spatial exposure densities.
- **Reporting Rule**: All benchmark tables and text in `paper/main.tex` and `FINAL_REPORT.md` must clearly reflect this exact distinction.
