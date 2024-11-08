#!/bin/bash

# Directory containing the files
DIRECTORY="/home/pcarragh/data/generations/webqa/yesno/"

# Loop through each file in the directory
for FILE in "$DIRECTORY"/*; do
  # Extract the base name of the file
  BASENAME=$(basename "$FILE")
  
  # Extract the last _X from the file name
  if [[ "$BASENAME" =~ (.*)_([0-9]+)(\.[^.]+)$ ]]; then
    PREFIX=${BASH_REMATCH[1]}
    X=${BASH_REMATCH[2]}
    SUFFIX=${BASH_REMATCH[3]}
    
    # Modify X according to the rules
    if (( X % 2 == 0 )); then
      NEW_X=$(( X / 2 ))
    else
      NEW_X=$(( (X - 1) / 2 ))
    fi
    
    # Construct the new file name
    NEWNAME="${PREFIX}_${NEW_X}${SUFFIX}"
    
    # Rename the file
    mv "$FILE" "$DIRECTORY/$NEWNAME"
  fi
done