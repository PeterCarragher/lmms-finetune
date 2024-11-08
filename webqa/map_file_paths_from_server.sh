#!/bin/bash

# Check if input JSON file is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_json_file>"
  exit 1
fi

JSON_FILE="$1"

# Define path mappings
declare -A path_mappings=(
  ["/data/nikitha/VQA_data/VQAv2/images/"]="/home/pcarragh/dev/webqa/images/images/"
  ["/data/nikitha/VQA_data/VQAv2/results/"]="/home/pcarragh/dev/webqa/segment/Inpaint-Anything/results/"
  ["/data/nikitha/VQA_data/results/old/bad_idx/webqa"]="/home/pcarragh/dev/webqa/image_gen/webqa/color/webqa"
  ["/data/nikitha/VQA_data/results/webqa_yesno"]="/home/pcarragh/dev/webqa/image_gen/webqa/yesno/webqa_yesno"
)

# Loop through each path mapping and replace it in the file
for old_path in "${!path_mappings[@]}"; do
  new_path="${path_mappings[$old_path]}"
  # Use sed to perform an in-place replacement for each path
  sed -i "s#${old_path}#${new_path}#g" "$JSON_FILE"
done

echo "Path replacement completed in $JSON_FILE"
