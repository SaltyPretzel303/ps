#!/bin/bash

if [ $# -lt 2 ]
then
	echo "Provide src file name as sa first argument ... "
	echo "Provide number of process to execute on as the second argument ..."
	exit 1
fi

full_path="$1"
just_file="$(basename $full_path)"
output_file="${just_file%.*}.out"
output_path="./bin/$output_file"

echo "Building: $full_path"
echo "Outputing to: $output_path"

mpicc "$full_path" -o "$output_path"

ret_code="$?"
if [ $ret_code -ne 0 ]
then
	echo "Build failed with error code: $ret_code"
	exit 1
fi

echo "Running on: $2 processes"
echo "..."

proc_num="$2"

mpiexec -n "$proc_num" "$output_path"