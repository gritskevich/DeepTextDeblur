#!/bin/bash

# Ensure the output directory for deblurred images exists
mkdir -p vb/deblured

# Use GNU Parallel to process 10 files concurrently.
# The {} placeholder represents the input file from vb/output,
# and {/} extracts the basename of the file.
parallel -j 10 'python run.py {} vb/deblured/{/}' ::: vb/output/*.png
