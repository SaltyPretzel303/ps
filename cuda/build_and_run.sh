#!/bin/bash

if [ $# -lt 1 ]
then
        echo "Provide src file name as the first argument ... "
        exit 1
fi


full_path="$1"
just_file="$(basename $full_path)"
output_file="${just_file%.*}.out"
output_path="./bin/$output_file"

echo -e "Building: \t $full_path"
echo -e "Outputing to: \t $output_path"

nvcc "$full_path" -o "$output_path"

ret_code="$?"

if [ $ret_code -ne 0 ]
then
        echo "Build failed with error code: $ret_code"
        exit 1
fi

echo "Running ... "
echo "=============================="

chmod a+x "$output_path"
./"$output_path"