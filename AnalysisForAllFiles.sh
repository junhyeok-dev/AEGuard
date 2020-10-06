#!/bin/sh

for f in ./dataset/cifar100/*.png
do
	echo "$f analyzing"
	python3 ImageAnalyzer.py "$f"
done
