#!/bin/bash


for value in {1..40}
do
	echo "running "
	echo $value
	python app.py
done
