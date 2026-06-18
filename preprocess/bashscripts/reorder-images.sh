#!/bin/sh
# ============================================================================
# Script: reorder-images.sh
#
# Purpose: Reorganize raw profilometer data from nested folders into flat structure
#
# Input: Raw data with nested folder structure:
#   rawdata/before/profilometer/x-x-x/*.png
#   rawdata/before/height/x-x-x/*.csv
#   rawdata/after/...
#
# Output: Flat structure with unique filenames:
#   processed/before/profilometer/x-x-x-y.png
#   processed/before/height/x-x-x-y.csv
#   etc.
#
# Usage: ./reorder-images.sh <source_dir> <target_dir>
# ============================================================================

copy_file() {
    """
    Copy a single file, determining its type and renaming with full sample ID.

    Detects whether file is before/after and what type (profilometer, height, etc),
    then copies with appropriate naming convention.
    """
    # Remove whitespace from path
    src=$(echo "$1" | tr -d '[:blank:]')

    # Enable case-insensitive pattern matching
    shopt -s nocasematch

    # Determine measurement time (before or after)
    if [[ "$src" =~ "before" ]]; then
        time="before"
    elif [[ "$src" =~ "after" ]]; then
        time="after"
    else
        echo "ERROR: $src has no determinable time (before/after)"
        return
    fi

    # Determine measurement type (data modality)
    if [[ "$src" =~ "profilometer" ]]; then
        type="profilometer"
    elif [[ "$src" =~ "raw" ]]; then
        type="height"
    elif [[ "$src" =~ "heatmap" ]]; then
        type="heatmap"
    elif [[ "$src" =~ "scanner" ]]; then
        type="scanner"
    else
        echo "ERROR: $src has no determinable type"
        return
    fi

    # Get original file name
    filename=$(basename "$src")

    # Copy file with appropriate naming
    if [ "$type" = "scanner" ]; then
        # Scanner files don't need sample ID appended
        target_path="$2/$time/$type/$filename"
        cp "$1" "$target_path"
    else
        # Extract sample ID from path (x-x-x format) and append to filename
        id="$(grep -Po '\d+-\d+-\d+' <<< "$src")"
        target_path="$2/$time/$type/$id-$filename"
        cp "$1" "$target_path"
    fi

    echo "Copied: $src → $target_path"
}


# Main execution
srcdir="$1"
targetdir="$2"

# Validate arguments
if [ -z "$srcdir" ] || [ -z "$targetdir" ]; then
    echo "Usage: $0 <source_directory> <target_directory>"
    echo ""
    echo "Reorganizes raw profilometer data from nested folders into flat structure"
    exit 1
fi

# Export function so it can be used by find
export -f copy_file

echo "Starting file reorganization..."
echo "Source: $srcdir"
echo "Target: $targetdir"

# For every file found in source, copy to the right folder with automatic naming
find "$srcdir" -type f -exec bash -c 'copy_file "$1" "$2"' _ {} "${targetdir}" \;

echo "✓ File reorganization complete!"
echo "  All files have been copied and renamed according to their type and measurement time."

