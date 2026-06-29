# Data Preprocessing Documentation

## Overview

This document describes the complete preprocessing pipeline used to prepare the training, validation, and test datasets for the U-Net corrosion prediction model. The preprocessing transforms raw profilometer measurements (RGB images and height profiles) into aligned, cleaned, and split datasets ready for model training.

The preprocessing is designed to be scientifically reproducible. All steps are documented with code implementations in Python notebooks and shell scripts, allowing others to verify and extend the preprocessing pipeline.

## Dataset Structure

The original dataset consists of before and after profilometer measurements of corroded samples:
- **Visual data**: RGB images from the profilometer scanner
- **Height data**: 2D height maps from the profilometer measurements
- **Inhibitor annotations**: Volume loss values and inhibitor types from post-processing analysis

## Preprocessing Pipeline

### Step 1: Data Quality Assessment and Manual Corrections

**Location**: `notebooks/check-data-quality.ipynb`

Before processing, the raw data is validated for quality issues and manually corrected where necessary.

#### Known Data Issues

1. **Image Naming Confusions**
   - Some before/after image pairs were mislabeled in the original dataset
   - Visual inspection revealed the following incorrect pairings:
     - `before/2-5-10 ↔ after/2-5-11` (actually should be `before/2-5-10 ↔ after/2-5-10` and `before/2-5-11 ↔ after/2-5-11`)
     - `before/3-8-3 ↔ after/3-8-6` (actually should be matched the opposite way)
   - **Resolution**: The after folders were renamed to align with the before folders for data consistency
   - **Note**: The connection to the actual corrosion inhibitor names remains ambiguous due to these confusions

2. **Duplicate and Missing Image Data**
   - Identical "before" images found: `before/4-6-11/3.png` = `before/4-6-10/3.png`
   - Visual inspection indicated that `after/4-6-10/3.png` had no valid corresponding before image
   - **Resolution**: Both `before/4-6-10/3.png` and `after/4-6-10/3.png` were removed from the dataset
   - **Rationale**: Cannot reliably compute volume loss without valid before-after pairs

3. **Missing Height Profile Data**
   - Sample `2-5-11-4` was missing its after-height profile
   - **Resolution**: This sample was removed from the dataset
   - **Rationale**: Height data is essential for computing pixel-wise volume loss

#### Data Quality Checks

The following quality checks are performed in the notebook:
- **Unmatched files**: Identifies before images without after counterparts and vice versa
- **Shape mismatches**: Detects images where height map dimensions don't match RGB image dimensions
- **Full dataset consistency**: Ensures all before/after/height combinations are aligned

### Step 2: Image Reorganization and Renaming

**Scripts**: `bashscripts/reorder-images.sh`

The original data is organized in nested folders by sample ID and measurement type. This step reorganizes all files into a flat hierarchy with unique identifiers.

**Transformations**:
- Before: `rawdata/before/profilometer/x-x-x/y.png`
- After: `rawdata/processed/before/profilometer/x-x-x-y.png`

**Implementation Details**:
- The script iterates through all files in the raw data directory
- For each file, it extracts the sample ID (e.g., `x-x-x`) and determines:
  - Measurement time (`before` or `after`)
  - Data type (`profilometer`, `height`, `heatmap`, or `scanner`)
  - Measurement index (image number)
- Files are copied to `rawdata/processed/` with new names: `{id}-{index}.{ext}`
- A copy log is maintained for traceability

**Rationale**: The flat structure simplifies subsequent processing steps and makes file operations more efficient.

### Step 3: Corrosion Inhibitor Metadata Processing

**Location**: `notebooks/get-volume-loss.ipynb`

The volume loss and corrosion inhibitor information is extracted from an Excel spreadsheet and converted to a machine-readable CSV format.

**Processing Steps**:
1. Read the Excel file (`rawdata/results/all-results.xlsx`)
2. Extract relevant columns: sample ID, inhibitor name, and volume loss measurements (4 measurements per sample)
3. Clean data:
   - Remove unwanted ID extensions
   - Replace malformed ID markers (`#` → `1`)
   - Remove line breaks in names
4. Reshape data from wide to long format:
   - Original format: one row per sample with 4 volume loss columns
   - New format: one row per image with corresponding volume loss
   - New ID format: `x-x-x-{1,2,3,4}` (sample base ID + image number)
5. Export to `data/volume-loss/volume-loss.csv`

**Output**: A CSV file with columns: `id`, `name` (inhibitor type), `volumeloss`

**Rationale**: The metric-like format enables easy lookup of inhibitor types during training and evaluation.

### Step 4: Sample Region of Interest Detection

**Location**: `notebooks/circle-fitting.ipynb`

The samples are circular and need to be isolated from the background for accurate alignment and analysis. This notebook automatically detects the sample boundaries using image processing techniques.

**Algorithm Overview**:
1. Extract sample patch: A larger patch from the center of the image
2. Extract background patches: Four smaller patches from the image corners
3. Histogram backprojection:
   - Compute histograms for sample and background patches
   - Backproject histograms onto the full image to identify sample vs. background
4. Binary map generation:
   - Threshold the backprojected images
   - Apply morphological operations (open/close) to clean the mask
5. Center detection:
   - Reflect and mirror the binary mask around candidate center points
   - Mutliply with the original mask and calculate mean
   - The point with highest correlation is the sample center
6. Radius estimation:
   - Invert the binary mask
   - Stepwise rotate the spherical binary mask around the center and multiply with itself
   - Multiply all rotated masks
   - The inner radius free of values determines the sample radius
7. Manual verification:
   - Detected circles are saved as images for visual inspection
   - Incorrectly detected circles are corrected manually
   - Final circle list saved to `rawdata/processed/before/sample-masking/circle-list/circle-list.csv`

**Why manual verification is necessary**: Automatic circle detection can fail due to:
- Low contrast between sample and background (especially for low corrosion)
- Shadows or reflections on the sample surface
- Irregular sample boundaries

**Output**: CSV file with detected circle parameters (center x, center y, radius) for each sample

### Step 5: Image Alignment and Masking

**Location**: `notebooks/match-before-after.ipynb`

Before and after images must be precisely aligned to compute accurate pixel-wise volume loss. Template matching is used to find the optimal spatial alignment.

#### Template Matching Procedure

**The Challenge**: Before and after profilometer measurements often have different:
- Image sizes (due to camera repositioning)
- Image shifts (misalignment between measurements)
- Slight distortions (sample warping or compression)

**Algorithm**:
1. **Initial coarse alignment**:
   - Center the before image using the detected circle boundaries
   - Apply 6 different template matching methods to find the after image offset:
     - `TM_CCOEFF`, `TM_CCOEFF_NORMED`, `TM_CCORR`, `TM_CCORR_NORMED`, `TM_SQDIFF`, `TM_SQDIFF_NORMED`
   - Use majority voting among the 6 methods to determine the most consistent offset

2. **Fine alignment with subpatches**:
   - If coarse alignment shows low consistency (≤0 agreement) the image might be distorted with respect to the befor image. In this case use the alignement that fits best on average.
   - Extract 4 subpatches from the sample image containing the border and background regions of the image from each corner
   - Template match each subpatch individually
   - Calculate the image offset from the average determined offsets from each patch

3. **Manual intervention for low-consistency matches**:
   - If automatic matching fails, manual intervention is required
   - Manually select image patches from high-contrast regions
   - Manually specify the offset for these samples
   - **Manually corrected samples**: `1-1-7-2`, `1-6-8-3`, `3-7-4-1`, `3-8-1-2`, `4-6-7-1`, `4-6-7-2`, `4-6-7-3`, `4-6-7-4`

**Known Limitations**:
- Template matching performs poorly when samples are slightly warped or deformed
- In real-world scenarios, markers on the samples would greatly improve alignment reliability
- Maintaining consistent camera distance and angle between measurements would reduce alignment artifacts

#### Image Overlay and Size Equalization

After determining the offset:
1. Offset the after image to overlay with before image
2. Pad the after image if the offset is negative
3. Crop the after image if the offset is positive
4. Equalize sizes between before and after by cropping to the intersection

#### Masking

Binary masks are created to identify valid regions for analysis:
1. Identify invalid pixels in images (all channels = 0, typically black background)
2. Identify invalid pixels in height profiles (NaN values, corrupted data)
3. Combine masks across both images and both height profiles

**Special cases documented**:
- Sample `3-8-3-3`: Height profile contains regions with NaN values; these are properly masked
- Sample `after/3-8-4-1`: Single corrupted height value; properly masked

### Step 6: Sample Centering

Still in `notebooks/match-before-after.ipynb`, after alignment, samples are centered:

1. Apply the mask to both before and after images
2. Extract the region of interest using the detected circle boundaries
3. Center the extracted region at the detected sample center
4. Save centered images and height maps

**Outputs**:
- `rawdata/processed/before/centered-image/{id}.png`
- `rawdata/processed/after/centered-image/{id}.png`
- `rawdata/processed/before/centered-height/{id}.npy`
- `rawdata/processed/after/centered-height/{id}.npy`

### Step 7: Binary Mask Creation

**Location**: `notebooks/create-mask.ipynb`

Binary masks indicate which pixels are valid (sample region) vs. invalid (background, corrupted data).

**Procedure**:
1. Read centered before and after images
2. Read centered before and after height profiles
3. Create mask where:
   - Before image has no valid data (all channels = 0)
   - After image has no valid data (all channels = 0)
   - Before height profile contains NaN values
   - After height profile contains NaN values
4. Apply mask to all data:
   - Set masked pixels in images to 0
   - Set masked pixels in height profiles to 0
5. Save mask for later use in data loading

**Outputs**:
- `rawdata/processed/before/image/{id}.png` (masked)
- `rawdata/processed/after/image/{id}.png` (masked)
- `rawdata/processed/before/height/{id}.npy` (masked)
- `rawdata/processed/after/height/{id}.npy` (masked)
- `rawdata/processed/mask/{id}.npy` (binary mask)

### Executing the Pipeline

To reproduce the preprocessing from raw data:

1. **Start with**: Raw profilometer data in original nested folder structure
2. **Run notebooks in order**:
   ```bash
   jupyter notebook notebooks/check-data-quality.ipynb
   # Manually correct issues identified

   jupyter notebook notebooks/circle-fitting.ipynb
   # Manually verify and correct detected circles

   jupyter notebook notebooks/match-before-after.ipynb
   # Manually correct low-consistency matches

   jupyter notebook notebooks/create-mask.ipynb

   jupyter notebook notebooks/split-data.ipynb

   jupyter notebook notebooks/get-volume-loss.ipynb

   jupyter notebook notebooks/calc-normlization-values.ipynb
   ```
3. **Or run bashscripts** for automated steps (after notebooks complete):
   ```bash
   bash bashscripts/reorder-images.sh ${input_dir} preprocess/rawdata/processed
   bash bashscripts/create-datafolder.sh
   bash bashscripts/copy-mask.sh
   ```

## Summary

The preprocessing pipeline transforms raw, noisy, and irregularly-aligned profilometer measurements into a clean, well-organized dataset suitable for machine learning. Key steps include data quality assessment, image alignment via template matching, sample region detection, masking, and splitting into train/validation/test sets. The entire pipeline is implemented in reproducible Python notebooks and shell scripts, enabling scientific reproducibility and future extensions.

Manual interventions are documented to ensure transparency and allow future researchers to understand and potentially improve the preprocessing methodology.
