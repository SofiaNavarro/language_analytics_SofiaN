#!/usr/bin/env bash

VENVNAME=ass3

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate

pip --version
pip install --upgrade pip
test -f requirements.txt && pip install -r requirements.txt

#download language pack
python -m spacy download en_core_web_sm

# make output folders
mkdir -p ../language_data/A3_output/

echo "build $VENVNAME"