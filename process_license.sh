#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p vb/license_output

# Find all .png files in vb/license and run the command in parallel
ls vb/license/*.png | parallel -j 0 python run.py {} vb/license_output/{/}