#!/bin/sh

for f in ./dataset/cifar100/org_002*.png
do
	echo "$f analyzing"
	python3 ImageAnalyzer.py "$f"
done
