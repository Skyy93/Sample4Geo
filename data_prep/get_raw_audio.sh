#!/bin/bash
# This requires GNU parallel and the `internetarchive` library to be installed on your system.
# You likely have the first one if you're running linux.
# For the second one, check https://archive.org/services/docs/api/internetarchive/installation.html
# Or simply run `pip install internetarchive`

mkdir -p data/raw_audio
cd data/raw_audio
tail -n +2 /Coding/Sample4Geo/data/metadata.csv | cut -d',' -f 1 | parallel -j8 ia download {} --glob="*.mp3"