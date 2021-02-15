#!/bin/sh

for f in ../dataset/train/0*.jpg
do
	echo "$f processing"
	python3 AEGenerator.py "$f"
done
