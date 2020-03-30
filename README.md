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
* Run the following command:
```	
python3 tools/train.py
```

## Evaluation
* Download cleargrasp-synthetic-val.zip from https://drive.google.com/drive/u/1/folders/1zEJodFoijmzjcD2Tpzl7EB7ftbGodhFS. After downloading and unzipping the folder, rename the folder from cleargrasp-synthetic-val to cleargrasp-testing-validation.
* Run the following command:
```
python3 tools/eval_cleargrasp_dataloader.py --resume_posenet='name_of_trained_model.pth' --dataset_root=your_root_dir

```
The above code stores the predicted and GT poses (in .mat format for each test sample) in the directory ./trained_models/cleargrasp/Densefusion_wo_refine_result. 

* Clone the repository https://github.com/cxt98/Pose_evaluation_toolbox for MATLAB evaluation
* Go to globals.m and change opt.root to your root directory. Also change opt.result_path to the result directory of python evaluation(./trained_models/cleargrasp/Densefusion_wo_refine_result). Also make sure the opt.classfile, opt.testlist_file and opt.test_image_path directories are correct
* Run evaluate_poses.m on MATLAB
* Run plot_accuracy.m on MATLAB for error plots
* Run show_poses_projection.m on MATLAB to visualize 2D projection of result poses on rgb images (currently results are not close to ground truth)

