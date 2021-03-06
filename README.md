# Digital_Pathology_Transfer_Learning
Implementation of "Combining weakly and strongly supervised learning improves strong supervision in Gleason pattern classification"

If you find this code useful, consider citing the accompanying article:
Sebastian Otálora, Niccolò Marini, Manfredo Atzori, and Henning Müller, et al. "Combining weakly and strongly supervised learning improves strong supervision in Gleason pattern classification" 

## Requirements
Python==3.6.9, tensorflow==1.14.0, numpy==1.17.3, opencv==4.2.0, pandas==0.25.2, pillow==6.1.0, keras==2.2.4, staintools==2.1.2

## Datasets
Two datasets are used for the experiments:
- [The Tissue Micro Array Zurich (TMAZ)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP)
- [The Cancer Genome Atlas-PRostate ADenocarcinoma (TCGA-PRAD)](https://portal.gdc.cancer.gov/projects/TCGA-PRAD) 

The lists of images used (split in training, validation, testing partitions) are in: 
- [TMAZ_images](https://github.com/ilmaro8/Digital_Pathology_Transfer_Learning/tree/main/csv_folder/partitions/)
- [TCGA_images](https://github.com/ilmaro8/Digital_Pathology_Transfer_Learning/tree/main/csv_folder/partitions/)

## Folder organization

### csv_files
It inclused the csv files. The csv files MUST BE in this format: filename, label.

### test
It includes the scripts to test fully-supervised, weakly-supervised, finetuning models at patch level (TMAZ), core level (TMAZ) and at whole slide image level (TCGA-PRAD).

### train
It includes the scripts to train fully-supervised, weakly-supervised, finetuning models.

### utils
It includes scripts used to extract patches from TMAs/WSIs and scrits to generate the csvs.


## Scripts organization
The repository includes the scripts for training the models (training_scripts), for testing the models (inference scripts) and for the preprocessing/generation of the csv files (utils).

### train:
- Fully_Supervised_training.py -n -p. The script is used to train the model with fully-supervision.
	* -n: number of experiment
	* -p: percentage of data to use (20,40,60,80,100).

- Weakly_Supervised_training.py -n The script is used to train the model with weakly-supervision.
	* -n: number of experiment. 

- Finetuning_with_weakly_annotated_data.py -n -p. The script is used to finetune the model (pre-trained with fully-supervision) with weakly-annotated data.
	* -n: number of experiment
	* -p: percentage of data used to pre-train (20,40,60,80,100).

### test:
- Inference_TMA_patches.py -n -e -p. The script is used to test the models at patch-level (TMAZ).
	* -e: method of train (options: fully/weak/finetune).
	* -n: number of experiment to test.
	* -p: percentage of data used to pre-train (20,40,60,80,100).

- Inference_TMA_cores.py -n -e -p. The script is used to test the models at core-level (TMAZ).
	* -e: method of train (options: fully/weak/finetune).
	* -n: number of experiment to test.
	* -p: percentage of data used to pre-train (20,40,60,80,100).

- Inference_TCGA_densely_keras.py -n -e -p. The script is used to test the models at WSI-level (TCGA-PRAD).
	* -e: method of train (options: fully/weak/finetune).
	* -n: number of experiment to test.
	* -p: percentage of data used to pre-train (20,40,60,80,100).

### utils
- TMA_Patch_Extractor.py -d -s -n -t -p. Script to extract the patches from the TMAZ dataset (using pixel-wise annotated masks).
	* -d: dataset where extract patches (train/valid/test).
	* -s: size of the tiles to extract (before the resize to 224x224).
	* -n: number of patches to extract.
	* -t: number of threads.
	* -p: minimum percentage of tissue in a tile to be accepted.

- WSI_Patch_Extractor.py -d -s -t -p. Script to extract the patches from the TCGA-PRAD dataset (using masks generated by [HistoQC](https://github.com/choosehappy/HistoQC)).
	* -d: dataset where extract patches (train/valid/test).
	* -s: size of the tiles to extract (before the resize to 224x224).
	* -t: number of threads.
	* -p: minimum percentage of tissue in a tile to be accepted.

- Create_percentage_datasets.py -n. Script to create the subset with percentage of data.
	* -n: number of experimet.

- Create_csv_weakly_annotated_data.py -d. Script to create the csv file with weakly-annotated data.
  * -d: partition (train/valid/test)

## Reference
If you find this repository useful in your research, please cite:

[1] Otálora, S., Marini, N., Müller, H., & Atzori, M. (2021). Combining weakly and strongly supervised learning improves strong supervision in Gleason pattern classification. BMC Medical Imaging, 21(1), 1-14.

Paper link: https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-021-00609-0

## Acknoledgements
This project has received funding from the EuropeanUnion’s Horizon 2020 research and innovation programme under grant agree-ment No. 825292 [ExaMode](http://www.examode.eu). Infrastructure fromthe SURFsara HPC center was used to train the CNN models in parallel. Otálora thanks Minciencias through the call 756 for PhD studies.
