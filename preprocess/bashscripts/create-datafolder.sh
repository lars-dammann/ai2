#!/bin/sh
cd "$(dirname "$0")"

valnum=25
testnum=25

cd ../rawdata/processed/before/centered-image

# Determine all filenames of the images and remove the last 6 characters to get the subfolder ID
shopt -s nullglob
fileNames=()
for file in *; do
    fileNames+=(${file::-6})
done

# Make the subfolder ids unique
uniqueNames=()
while IFS= read -r -d '' x
do
    uniqueNames+=("$x")
done < <(printf "%s\0" "${fileNames[@]}" | sort -uz)

# Draw a $valTestNum number of random samples from subfolder ids
valTestNum=$(($valnum+$testnum))
valTestArray=($(printf "%s\n" "${uniqueNames[@]}" | shuf -n $valTestNum))

# Remove drawn random subfolder ids from list of all ids
trainArray=($(echo "${uniqueNames[@]}" "${valTestArray[@]}" | tr ' ' '\n' | sort | uniq -u))

# Make the first $valnum elements the validation folder data
valArray=("${valTestArray[@]:0:$valnum}")
# Determine the differene between validation folder and random drawn samples to determine the test set data
testArray=($(echo "${valTestArray[@]}" "${valArray[@]}" | tr ' ' '\n' | sort | uniq -u))

# Make sure there is no data leakage
intersectValTest=($(comm -12 <(printf '%s\n' "${valArray[@]}" | LC_ALL=C sort) <(printf '%s\n' "${testArray[@]}" | LC_ALL=C sort)))
intersectValTrain=($(comm -12 <(printf '%s\n' "${valArray[@]}" | LC_ALL=C sort) <(printf '%s\n' "${trainArray[@]}" | LC_ALL=C sort)))
intersectTrainTest=($(comm -12 <(printf '%s\n' "${trainArray[@]}" | LC_ALL=C sort) <(printf '%s\n' "${testArray[@]}" | LC_ALL=C sort)))
if [ -z "$intersectValTest" ] && [ -z "$intersectValTrain" ] && [ -z "$intersectTrainTest" ]
then
    :
else
    echo "Train val and test files not unique"
    exit 1
fi

cd ../..

shopt -s globstar
# Copy train files
for id in ${trainArray[@]}
do
    cp before/centered-image/$id-* "../../../data/train/before/image/."
    cp before/centered-height/$id-* "../../../data/train/before/height/."
    cp after/centered-image/$id-* "../../../data/train/after/image/."
    cp after/centered-height/$id-* "../../../data/train/after/height/."
done

# Copy val files
for id in ${valArray[@]}
do
    cp before/centered-image/$id-* "../../../data/val/before/image/."
    cp before/centered-height/$id-* "../../../data/val/before/height/."
    cp after/centered-image/$id-* "../../../data/val/after/image/."
    cp after/centered-height/$id-* "../../../data/val/after/height/."
done

# Copy test files
for id in ${testArray[@]}
do
    cp before/centered-image/$id-* "../../../data/test/before/image/."
    cp before/centered-height/$id-* "../../../data/test/before/height/."
    cp after/centered-image/$id-* "../../../data/test/after/image/."
    cp after/centered-height/$id-* "../../../data/test/after/height/."
done