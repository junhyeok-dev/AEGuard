#!/bin/sh

for f in ./dataset/cifar100/adv_002*.png
do
	echo "$f analyzing"
	python3 ImageAnalyzer.py "$f"
done
