#!/bin/sh

for f in ./dataset/cifar100/adv_*.png
do
	echo "$f analyzing"
	python3 ImageAnalyzer.py "$f"
done
