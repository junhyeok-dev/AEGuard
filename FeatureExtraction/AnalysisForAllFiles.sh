#!/bin/sh

for f in ../AEGenerator/adv*.png
do
	echo "$f analyzing"
	python3 ImageAnalyzer.py "$f"
done
