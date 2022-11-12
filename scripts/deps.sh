#!/bin/bash

deps() {
  pip install -r ./requirements.txt
}

__deps__info() {
  echo " - \`deps\` install dependencies specified in requirements.txt use"
}
