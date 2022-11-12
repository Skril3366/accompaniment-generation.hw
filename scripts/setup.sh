#!/bin/bash

source ./env/bin/activate
source ./scripts/run.sh

# Install dependencies
echo "Installing dependencies specified in requirements.txt"
pip install -r requirements.txt

echo
echo "All the dependencies are installed"
echo "To run the application use \`run\` command with corresponding arguments"
