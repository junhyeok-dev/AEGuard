#!/bin/sh

for f in ../AEGenerator/org_001*.png
do
	echo "$f analyzing"
	python3 ImageAnalyzer.py "$f"
done
