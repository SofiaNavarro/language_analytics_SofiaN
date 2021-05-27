#!/usr/bin/env bash

VENVNAME=ass4

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate

pip --version
pip install --upgrade pip
test -f requirements.txt && pip install -r requirements.txt

python -m spacy download en_core_web_sm

# make output folders
mkdir -p ../language_data/A4_output/

echo "build $VENVNAME"
