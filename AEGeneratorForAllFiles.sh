#!/bin/sh

for f in *.jpg
do
	echo "$f processing"
	python3 AEGenerator.py "$f"
done
