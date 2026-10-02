# ResQ-AI Data Card

## Dataset Overview
- **Name**: ResQ-AI Multi-Modal Flood Dataset
- **Version**: 1.0
- **Date**: October 2026
- **License**: Various (see individual datasets)

## Component Datasets

### Sen1Floods11
- **Source**: Google Cloud Storage (gs://sen1floods11)
- **Citation**: Bonafilia et al. (2020)
- **License**: CC-BY-4.0
- **Content**: 446 hand-labeled 512x512 chips from 14 flood events
- **Modalities**: Sentinel-1 SAR (VV, VH), Sentinel-2 optical (13 bands)
- **Labels**: Binary flood/non-flood per pixel
- **Splits**: Official train/valid/test + Bolivia holdout
- **Known Issues**: Class imbalance, cloud contamination in optical

### Copernicus GLO-30 DEM
- **Source**: AWS Open Data
- **Resolution**: 30m
- **Derived features**: Elevation, slope, HAND, TWI, distance to drainage

### ESA WorldCover
- **Source**: ESA
- **Resolution**: 10m
- **Classes**: 11 land cover types

### CHIRPS
- **Source**: UCSB Climate Hazards Group
- **Resolution**: 0.05 degrees daily
- **Usage**: 30-day antecedent rainfall sequence

### WorldPop
- **Source**: WorldPop Hub
- **Resolution**: 100m
- **Usage**: Population exposure estimation

### OpenStreetMap
- **Source**: OSM via osmnx
- **Usage**: Road networks, buildings, facilities

## Preprocessing
- All rasters aligned to 10m chip grid
- Per-modality normalization
- NoData handling
- Spatial and event-level splitting to avoid leakage

## Ethical Considerations
- Population data used only for aggregate impact estimation
- No personally identifiable information
- Models should not be deployed without validation on target region

## Limitations
- Sen1Floods11 covers limited geographic diversity
- Temporal mismatch between datasets possible
- Building footprint completeness varies by region
