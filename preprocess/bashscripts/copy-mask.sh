#!/bin/sh
# ============================================================================
# Script: copy-mask.sh
#
# Purpose: Copy binary masks to corresponding train/val/test directories
#
# Background:
#   Binary masks are created during preprocessing (create-mask.ipynb) and stored
#   in rawdata/processed/mask/. After data splitting, masks must be copied to
#   the corresponding split directories for use during model training.
#
#   Masks identify:
#   - Valid pixels (sample region)
#   - Invalid pixels (background, corrupted data)
#
#   Masks are used during data loading to compute loss only on valid pixels
#   and ensure proper normalization statistics.
#
# Functionality:
#   For each split (train, val, test):
#     For each image in the split:
#       Copy corresponding mask from rawdata/processed/mask/
#       to data/{split}/mask/
#
# ============================================================================

cd "$(dirname "$0")"
cd ..

echo "Copying binary masks to train/val/test directories..."

# For each data split, copy corresponding masks
for data in train val test; do
    echo "  Processing $data split..."

    # For each before/after time point
    for time in before after; do
        # For each image file in the split
        for file in ../data/$data/$time/image/*; do
            # Extract the base filename (without path)
            basefile=$(basename "$file")

            # Remove the file extension to get the sample ID
            sample_id="${basefile%.*}"

            # Copy the corresponding mask from rawdata
            # The mask file should have same base name but .npy extension
            cp "rawdata/processed/mask/${sample_id}.npy" "../data/$data/$time/mask/." 2>/dev/null
        done
    done
done

echo "✓ Mask copying complete!"
echo "  All masks have been copied to their corresponding split directories."
