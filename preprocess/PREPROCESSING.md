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
   - Rotate the binary mask around candidate center points
   - Correlate with the original mask
   - The point with highest correlation is the sample center
6. Radius estimation:
   - For each angle around the detected center, find where the mask transitions from background to sample
   - Average across all angles to estimate the radius
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
   - If coarse alignment shows low consistency (≤0 agreement), use statistical template matching
   - Extract 5 subpatches from the centered before image:
     - Two corner regions (top-left and bottom-right quadrants, excluding center)
     - One central region (center quadrant)
   - Template match each subpatch individually
   - Calculate precision as the mean radial distance of subpatch matches from the coarse match
   - Calculate consistency as the number of subpatches agreeing within a distance limit

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
4. Apply morphological operations to clean the masks

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

### Step 8: Data Splitting

**Location**: `notebooks/split-data.ipynb` and `bashscripts/create-datafolder.sh`

The processed data is split into training, validation, and test sets while maintaining data integrity and preventing data leakage.

#### Splitting Strategy

**Objective**: Test the model's ability to generalize to unseen corrosion inhibitors while ensuring proper train/val/test separation.

**Procedure**:
1. Load all sample IDs and their corresponding inhibitor types from `volume-loss.csv`
2. Separate NaCl samples (these are put into training set)
3. From remaining samples:
   - Randomize order (seed=5 for reproducibility)
   - Allocate 25 samples to validation set
   - Allocate 25 samples to test set
   - Remaining samples go to training set
4. Add NaCl samples to training set
5. For each sample ID, copy all 4 associated images (1-4) to the appropriate set

#### Data Leakage Prevention

The splitting logic explicitly checks that:
- No validation sample appears in training set
- No validation sample appears in test set
- No training sample appears in test set

This is critical because each sample ID has 4 associated images, and we must ensure all 4 are in the same split.

#### Final Dataset Composition

- **Training set**: 161 sample IDs = 644 images
  - 36 NaCl samples
  - 125 other inhibitor types
- **Validation set**: 25 sample IDs = 100 images
- **Test set**: 25 sample IDs = 100 images

**Note on inhibitor consistency**: Due to the image naming confusions (see Step 1), the mapping between samples and inhibitor types may contain errors. Specifically:
- Sample `3-8-6` is listed as NaCl
- Sample `3-8-3` was confused with `3-8-6`
- Therefore, the actual inhibitor type of sample `3-8-3` is uncertain

#### Copy Using Bashscript

**Script**: `bashscripts/create-datafolder.sh`

This script implements the splitting procedure:
1. Reads all sample IDs from `rawdata/processed/before/centered-image/`
2. Extracts unique base IDs (e.g., `x-x-x`)
3. Randomly draws 50 base IDs for validation/test
4. Remaining base IDs go to training
5. Splits the 50 into 25 validation and 25 test
6. Validates no data leakage
7. Copies centered images and heights to `data/{train,val,test}/{before,after}/{image,height}/`

### Step 9: Mask Distribution

**Script**: `bashscripts/copy-mask.sh`

After data splitting, binary masks must be copied to the corresponding data split folders:

**Procedure**:
1. For each data split (train, val, test)
2. For each time point (before, after)
3. For each image file in the split
4. Copy the corresponding mask from `rawdata/processed/mask/` to `data/{split}/{time}/mask/`

**Rationale**: Masks are stored separately but referenced during data loading to identify valid pixels for loss computation.

### Step 10: Normalization Calculation

**Location**: `notebooks/calc-normlization-values.ipynb`

To ensure stable training, height profiles are normalized using statistics computed from the training set.

**Process**:
1. Load all training set height profiles
2. Compute per-channel statistics:
   - Mean value
   - Standard deviation
3. Save normalization parameters to a file or config
4. These parameters are used during data loading to normalize all height profiles

**Rationale**: Normalizing to zero mean and unit variance improves neural network training stability and convergence.

## Reproducibility and File Structure

### Key Files for Reproducibility

```
preprocess/
├── PREPROCESSING.md                         # This documentation
├── notebooks/
│   ├── check-data-quality.ipynb             # Step 1: Quality assessment
│   ├── circle-fitting.ipynb                 # Step 4: Region detection
│   ├── match-before-after.ipynb             # Step 5-6: Alignment & centering
│   ├── create-mask.ipynb                    # Step 7: Masking
│   ├── split-data.ipynb                     # Step 8: Data splitting
│   ├── get-volume-loss.ipynb                # Step 3: Metadata processing
│   └── calc-normlization-values.ipynb       # Step 10: Normalization
├── bashscripts/
│   ├── reorder-images.sh                    # Step 2: Reorganization
│   ├── create-datafolder.sh                 # Step 8: Splitting
│   └── copy-mask.sh                         # Step 9: Mask distribution
└── rawdata/
    ├── results/all-results.xlsx             # Input: Volume loss data
    └── processed/                           # Intermediate outputs (DO NOT MODIFY)
```

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

### Variables That Can Be Tuned

If you want to modify the preprocessing, the following parameters can be adjusted:

**Data splitting** (`create-datafolder.sh`, `split-data.ipynb`):
- `valnum = 25`: Number of validation samples
- `testnum = 25`: Number of test samples
- `random_seed = 5`: Random seed for reproducible splitting

**Circle fitting** (`circle-fitting.ipynb`):
- Histogram backprojection thresholds
- Morphological operation kernel sizes
- Search ranges for center and radius detection

**Template matching** (`match-before-after.ipynb`):
- Distance limit for consistency checking
- Number and positions of subpatches
- Image masking parameters before alignment

## Known Limitations and Future Improvements

### Current Limitations

1. **Image Alignment**: While template matching works for most samples, slightly warped or deformed samples have imperfect pixel-level alignment
   - **Impact**: May introduce small errors in volume loss computation for severely corroded samples
   - **Potential Solution**: Add fiducial markers to samples, maintain precise camera calibration

2. **Height Profile Alignment**: Height profiles may have slight shifts relative to RGB images
   - **Impact**: Could cause height-image misalignment in rare cases
   - **Potential Solution**: Ensure measurement hardware synchronizes image and height acquisitions

3. **Semi-Automatic Circle Detection**: Requires manual verification and correction
   - **Impact**: Time-consuming for large datasets
   - **Potential Solution**: Train a machine learning model for circle detection

4. **Manual Intervention**: Some samples required manual handling
   - **Impact**: Potential source of inconsistency if future users make different choices
   - **Potential Solution**: Document exact manual decisions in code rather than comments

5. **Inhibitor Label Ambiguity**: Image naming confusions make some inhibitor labels uncertain
   - **Impact**: Model evaluation with respect to inhibitor generalization may be affected
   - **Potential Solution**: Verify inhibitor labels with experimental documentation

### Recommendations for Future Data Collection

1. **Sample Markers**: Add visible fiducial markers (dots, cross-hairs) to improve alignment
2. **Consistent Camera Position**: Maintain precise camera distance and angle between before/after measurements
3. **Data Validation**: Create checksums or hashes to verify data integrity during transfer
4. **Metadata Logging**: Automatically record measurement parameters (camera settings, calibration, date) with data
5. **Quality Thresholds**: Define acceptance criteria for image quality before measurement completion

## References

For additional context on the data and corrosion measurement protocol, refer to:
- Raw data documentation (if available in `rawdata/`)
- Experimental design documents
- Profilometer manual and calibration procedures

## Summary

The preprocessing pipeline transforms raw, noisy, and irregularly-aligned profilometer measurements into a clean, well-organized dataset suitable for machine learning. Key steps include data quality assessment, image alignment via template matching, sample region detection, masking, and splitting into train/validation/test sets. The entire pipeline is implemented in reproducible Python notebooks and shell scripts, enabling scientific reproducibility and future extensions.

Manual interventions are documented to ensure transparency and allow future researchers to understand and potentially improve the preprocessing methodology.
