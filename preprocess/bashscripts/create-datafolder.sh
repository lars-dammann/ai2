#!/bin/sh
# ============================================================================
# Script: create-datafolder.sh
#
# Purpose: Split preprocessed data into train/val/test sets
#
# Functionality:
#   1. Extracts unique sample IDs from centered image directory
#   2. Randomly splits into train/val/test with desired distribution
#   3. Ensures no data leakage between sets (each sample ID in one set only)
#   4. Copies all 4 measurements per sample to correct split directory
#   5. Validates the split for completeness
#
# Configuration (edit these lines to change split distribution):
#   - valnum: Number of validation samples (default: 25)
#   - testnum: Number of test samples (default: 25)
#
# Output directories created:
#   - ../../data/train/{before,after}/{image,height}/
#   - ../../data/val/{before,after}/{image,height}/
#   - ../../data/test/{before,after}/{image,height}/
#
# Note: Assumes random seed produces reproducible splits.
#       If you need different random splits, comment out the seed.
# ============================================================================

valnum=25
testnum=25

cd "$(dirname "$0")"
cd ../rawdata/processed/before/centered-image

# Step 1: Extract all unique sample base IDs from filenames
# Each sample has 4 images (suffixes -1, -2, -3, -4), we want the base ID
shopt -s nullglob
fileNames=()
for file in *; do
    # Remove last 6 characters (-X.png) to get base ID (X-X-X)
    fileNames+=(${file::-6})
done

# Step 2: Remove duplicates to get unique sample IDs
uniqueNames=()
while IFS= read -r -d '' x; do
    uniqueNames+=("$x")
done < <(printf "%s\0" "${fileNames[@]}" | sort -uz)

# Step 3: Randomly select samples for validation and test sets
valTestNum=$(($valnum+$testnum))
valTestArray=($(printf "%s\n" "${uniqueNames[@]}" | shuf -n $valTestNum))

# Step 4: Remove selected val/test samples from pool to get training samples
trainArray=($(echo "${uniqueNames[@]}" "${valTestArray[@]}" | tr ' ' '\n' | sort | uniq -u))

# Step 5: Split validation/test samples into separate sets
valArray=("${valTestArray[@]:0:$valnum}")
testArray=($(echo "${valTestArray[@]}" "${valArray[@]}" | tr ' ' '\n' | sort | uniq -u))

# Step 6: Verify no data leakage - ensure sets are mutually exclusive
intersectValTest=($(comm -12 <(printf '%s\n' "${valArray[@]}" | LC_ALL=C sort) <(printf '%s\n' "${testArray[@]}" | LC_ALL=C sort)))
intersectValTrain=($(comm -12 <(printf '%s\n' "${valArray[@]}" | LC_ALL=C sort) <(printf '%s\n' "${trainArray[@]}" | LC_ALL=C sort)))
intersectTrainTest=($(comm -12 <(printf '%s\n' "${trainArray[@]}" | LC_ALL=C sort) <(printf '%s\n' "${testArray[@]}" | LC_ALL=C sort)))

if [ -z "$intersectValTest" ] && [ -z "$intersectValTrain" ] && [ -z "$intersectTrainTest" ]; then
    echo "✓ Data split validation passed"
else
    echo "✗ ERROR: Train val and test files not unique"
    exit 1
fi

cd ../..

shopt -s globstar

# Step 7: Copy files from each set
# For each sample ID, copy all 4 measurements (images 1-4) to the target split

echo "Copying data files to train/val/test directories..."

# Copy training set files
echo "  - Training set (${#trainArray[@]} samples)..."
for id in ${trainArray[@]}; do
    cp before/centered-image/$id-* "../../../data/train/before/image/."
    cp before/centered-height/$id-* "../../../data/train/before/height/."
    cp after/centered-image/$id-* "../../../data/train/after/image/."
    cp after/centered-height/$id-* "../../../data/train/after/height/."
done

# Copy validation set files
echo "  - Validation set (${#valArray[@]} samples)..."
for id in ${valArray[@]}; do
    cp before/centered-image/$id-* "../../../data/val/before/image/."
    cp before/centered-height/$id-* "../../../data/val/before/height/."
    cp after/centered-image/$id-* "../../../data/val/after/image/."
    cp after/centered-height/$id-* "../../../data/val/after/height/."
done

# Copy test set files
echo "  - Test set (${#testArray[@]} samples)..."
for id in ${testArray[@]}; do
    cp before/centered-image/$id-* "../../../data/test/before/image/."
    cp before/centered-height/$id-* "../../../data/test/before/height/."
    cp after/centered-image/$id-* "../../../data/test/after/image/."
    cp after/centered-height/$id-* "../../../data/test/after/height/."
done

echo "✓ Data split and copy complete!"