#!/bin/bash

# This is a script which copies the contents of the tutorials
# directory into a build directory (_build/ipynbs).
# The notebook markdown files are converted to ipynbs with other
# files (non-executable md files, static images, etc) are copied
# directly.
# This is intended for jupyterlite to build pointing to the build
# directory since the markdown files do not currently work in 
# jupyterlite.

# Find Markdown files convert.
files_to_process=$(find docs/source/tutorials -type f)

OUTDIR="build/ipynbs"

# Identify Markdown files that are Jupytext and convert them all.
for file in ${files_to_process}; do
    # Ensure result directory exists
    echo "Making directory: $OUTDIR/$(dirname $file)"
    mkdir -p $OUTDIR/$(dirname $file)

    echo loop in $file
    # Extract the kernel information from the Jupytext Markdown file.
    kernel_info=$(grep -A 10 '^---$' "$file" | grep -E 'kernelspec')
    # Copy directly if not a notebook file
    if [ -z "$kernel_info" ]; then
      cp $file $OUTDIR/$file
      continue
    fi
    # Convert to ipynb format, to be consumed by pytest nbval plugin.
    notebook_file="${file%.md}.ipynb"
    jupytext --to ipynb "$file" --output $OUTDIR/${notebook_file}
done