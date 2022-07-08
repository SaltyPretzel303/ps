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

echo -e "Building: \t $full_path"
echo -e "Outputing to: \t $output_path"

# lm will link libraries (for example so that math.h can be used)_
mpic++ "$full_path" -o "$output_path" -lm

ret_code="$?"
if [ $ret_code -ne 0 ]
then
	echo "Build failed with error code: $ret_code"
	exit 1
fi

echo -e "Running on: \t $2 processes"
echo "..."

proc_num="$2"

# mpiexec -n "$proc_num" --hostfile ./hostfile "$output_path"
mpiexec -n "$proc_num" "$output_path"
