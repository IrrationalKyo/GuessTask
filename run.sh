#!/bin/bash

source /home/rick/CS/tensorflow/bin/activate

for value in {1..40}
do
	echo "running "
	echo $value
	python app.py
done
