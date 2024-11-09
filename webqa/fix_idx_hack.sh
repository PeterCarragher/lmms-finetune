#!/bin/bash

# Directory containing the files
DIRECTORY="/data/nikitha/VQA_data/results/webqa_yesno/"

# Loop through each file in the directory
for FILE in "$DIRECTORY"/*; do
  # Extract the base name of the file
  BASENAME=$(basename "$FILE")
  
  # Remove the last _X from the file name
  NEWNAME=$(echo "$BASENAME" | sed 's/_[0-9]\+\(\.[^.]*\)$/\1/')
  
  # Rename the file
  mv "$FILE" "$DIRECTORY/$NEWNAME"
done