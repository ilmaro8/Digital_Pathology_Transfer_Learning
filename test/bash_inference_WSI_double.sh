#!/bin/bash
for (( X=0; X<10; X++ ))
do
	python Inference_TCGA_densely_keras_double_output.py -n $X -p weak -s 100
done

for (( X=0; X<9; X++ ))
do
	python Inference_TCGA_densely_keras_double_output.py -n $X -p weak_fine -s 100
done

for (( X=0; X<9; X++ ))
do
	python Inference_TCGA_densely_keras_double_output.py -n $X -p weak_fine -s 80
done

for (( X=0; X<9; X++ ))
do
	python Inference_TCGA_densely_keras_double_output.py -n $X -p weak_fine -s 60
done

for (( X=0; X<9; X++ ))
do
	python Inference_TCGA_densely_keras_double_output.py -n $X -p weak_fine -s 40
done

for (( X=0; X<9; X++ ))
do
	python Inference_TCGA_densely_keras_double_output.py -n $X -p weak_fine -s 20
done