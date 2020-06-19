#!/bin/bash

# exit if a command fails
set -e

# folders to build
declare -a dirs=("hysteresis/Plugin")

for dir in "${dirs[@]}"; do
    echo "Testing ${dir}..."
    (
        cd $dir
        if [[ ! -d build ]]; then mkdir build; fi
        cd build/

        cmake ..
        cmake --build . --config Release
    )
done
