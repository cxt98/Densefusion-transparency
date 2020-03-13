# DenseFusion

## Requirements

* Python 2.7/3.5/3.6 (If you want to use Python2.7 to run this repo, please rebuild the `lib/knn/` (with PyTorch 0.4.1).)
* [PyTorch 0.4.1](https://pytorch.org/) ([PyTroch 1.0 branch](<https://github.com/j96w/DenseFusion/tree/Pytorch-1.0>))
* PIL
* scipy
* numpy
* pyyaml
* logging
* matplotlib
* CUDA 7.5/8.0/9.0 (Required. CPU-only will lead to extreme slow training speed because of the loss calculation of the symmetry objects (pixel-wise nearest neighbour loss).)

## Datasets

Download the ClearGrasp training and testing data from https://sites.google.com/view/cleargrasp/data?authuser=0. The folders where this data is stored will be the dataset_root argument in train.py

## Training

* Create a sub folder cleargrasp in trained_models/ directory
* Download the meta-files.zip from https://drive.google.com/drive/u/1/folders/1zEJodFoijmzjcD2Tpzl7EB7ftbGodhFS. Storing these meta-files should be as follows:
	*  cleargrasp-dataset-train/
		* cup-with-waves-train
			* meta-files
				* all the meta files from unzipping meta-files_cup-with-waves.zip
* Run the below command:
```	
python3 tools/train.py
```

