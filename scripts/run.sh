#!/bin/bash

run() {
	python ./src/main.py "$@"
}

__run__info(){
  echo " - \`run\` run command with corresponding arguments"
}
