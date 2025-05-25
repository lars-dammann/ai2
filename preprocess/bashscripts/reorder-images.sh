#!/bin/sh

copy_file () {
    # Remove blanks from path
    src=$(echo $1 | tr -d '[:blank:]')
    
    # Ignore cases in regex match
    shopt -s nocasematch

    # Figure out if image was created before or after cleaning
    if [[ "$src" =~ "before" ]]; then
        time="before"
    elif [[ "$src" =~ "after" ]]; then
        time="after"
    else
        echo "$src has no determinable time"
        return
    fi

    # Figure out type of image
    if [[ "$src" =~ "profilometer" ]]; then
        type="profilometer"
    elif [[ "$src" =~ "raw" ]]; then
        type="height"
    elif [[ "$src" =~ "heatmap" ]]; then
        type="heatmap"
    elif [[ "$src" =~ "scanner" ]]; then
        type="scanner"
    else
        echo "$src has no determinable type"
        return
    fi

    # Get original file name
    filename=$(basename "$src")
    
    # If image type is scanner no id needs to be added in filename
    if [ $type == "scanner" ]; then
        echo "Save $src to $2/$time/$type/$filename"
        echo "$src $2/$time/$type/$filename" >> "$2/copy-log.txt"
        cp "$1" "$2/$time/$type/$filename"
    else
        # Else create new filename with full id
        id="$(grep -Po '\d+-\d+-\d+' <<< "$src")"
        echo "Save $src to $2/$time/$type/$id-$filename"
        echo "$src $2/$time/$type/$id-$filename" >> "$2/copy-log.txt"
        cp "$1" "$2/$time/$type/$id-$filename"
    fi
}


srcdir=$1
targetdir=$2

# Export function so it is usable by find
export -f copy_file

# For every file found, copy to the right folder
find $srcdir -type f -exec bash -c "copy_file \"{}\" $targetdir" \;

