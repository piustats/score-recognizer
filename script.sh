#!/bin/sh
set -e

python3 main.py predict --disable-logs $CNN_MODEL $TESSERACT_MODEL $@