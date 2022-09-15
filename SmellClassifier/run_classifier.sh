#!/bin/bash

echo "Running the classifier."

python smell_classifier.py --model_path "model-path" --input_path "input_folder" \
                           --output_path "output" --lang "any-project-language"
