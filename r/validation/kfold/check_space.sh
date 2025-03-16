#!/bin/bash

# Loop through each directory in the current directory
for dir in */ .*/ ; do
    # Use find to count all files in the directory and its subdirectories
    file_count=$(find "$dir" -type f | wc -l)

    # Print the directory name and the file count
    echo "$dir: $file_count"
done