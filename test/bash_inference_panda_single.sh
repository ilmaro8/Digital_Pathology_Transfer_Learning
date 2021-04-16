#!/bin/bash
for (( X=0; X<10; X++ ))
do
	python Inference_panda_densely_keras.py -n $X -p fully -s 100
done

for (( X=0; X<10; X++ ))
do
	python Inference_panda_densely_keras.py -n $X -p fully -s 80
done

for (( X=0; X<10; X++ ))
do
	python Inference_panda_densely_keras.py -n $X -p fully -s 60
done

for (( X=0; X<10; X++ ))
do
	python Inference_panda_densely_keras.py -n $X -p fully -s 40
done

for (( X=0; X<10; X++ ))
do
	python Inference_panda_densely_keras.py -n $X -p fully -s 20
done