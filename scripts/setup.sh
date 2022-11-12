#!/bin/bash

source ./env/bin/activate
source ./scripts/run.sh
source ./scripts/deps.sh

cat ./scripts/name.txt

echo "The following commands are added on top standard python env commands:"
__run__info
__deps__info
