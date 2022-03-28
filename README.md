

# MatDeepLearn_DOS

This repo contains the code for predicting density of states (DOS) using a modified version of the MatDeepLearn code, as described in the paper "Physically informed machine learning prediction of electronic density of states"

Please <a href="#roadmap">contact</a> the developer(s) for bug fixes and feature requests.

## Table of contents
<ol>
	<li><a href="#installation">Installation</a></li>
	<li><a href="#usage">Usage</a></li>
	<li><a href="#license">License</a></li>
	<li><a href="#acknowledgements">Acknowledgements</a></li>
</ol>

## Installation


### Prerequisites

Prerequisites are listed in requirements.txt. You will need two key packages, 1. Pytorch and 2. Pytorch-Geometric. You may want to create a virtual environment first, using Conda for example.

1. **Pytorch**: The package has been tested on Pytorch 1.8. To install, for example:
	```bash
	pip install torch==1.8.0 torchvision==0.9.0
	```
2. **Pytorch-Geometric:**  The package has been tested on Pytorch-Geometric. 1.7.0. To install, [follow their instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html), for example:
	```bash
    pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
    pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
    pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
    pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
    pip install torch-geometric
	```	
    where where ${CUDA} and ${TORCH} should be replaced by your specific CUDA version (cpu, cu92, cu101, cu102, cu110, cu111) and PyTorch version (1.4.0, 1.5.0, 1.6.0, 1.7.0, 1.8.0), respectively.

3. **Remaining requirements:** The remainder may be installed by:
	```bash
    git clone https://github.com/vxfung/MatDeepLearn
    cd MatDeepLearn    
	pip install -r requirements.txt
	```
    
## Usage

### Running a calculation

1. Download the datasets as described in MatDeepLearn/data/ :
	
2.	To run the code, type:
	```bash
	python main.py --data_path='your_path_here' --model='your_model_here'
	```
	where the settings will be read from the provided config.yml.
	
3. The program will begin training; on a regular CPU this should take ~10-20s per epoch. It is recommended to use GPUs which can provide a roughly ~5-20 times speedup, which is needed for the larger datasets. As default, the program will provide two outputs: (1) "my_model.pth" which is a saved model which can be used for predictions on new structures, (2) "myjob_train_job_XXX_outputs.csv" where XXX are train, val and test; these contain structure ids, targets and the predicted values from the last epoch of training and validation, and for the test set.

### The configuration file

The configuration file is provided in .yml format and encodes all the settings used. By default it should be in the same directory as main.py or specified in a separate location by --config_path in the command line. 

There are four categories or sections: 1. Job, 2. Processing, 3. Training, 4. Models

1. **Job:** This section encodes the settings specific to the type of job to run. Current supported are: Training, Predict, Repeat, CV, Hyperparameter, Ensemble, Analysis. The program will only read the section for the current job, which is selected by --run_mode in the command line, e.g. --run_mode=Training. Some other settings which can be changed in the command line are: --job_name, --model, --seed, --parallel.

2. **Processing:** This section encodes the settings specific to the processing of structures to graphs or other features. Primary settings are the "graph_max_radius", "graph_max_neighbors" and "graph_edge_length" which controls radius cutoff for edges, maximum number of edges, and length of edges from a basis expansion, respectively. Prior to this, the directory path containing the structure files must be specified by "data_path" in the file or --data_path in the command line.

3. **Training:** This section encodes the settings specific to the training. Primary settings are the "loss", "train_ratio" and "val_ratio" and "test_ratio". This can also be specified in the command line by --train_ratio, --val_ratio, --test_ratio.

4. **Models:** This section encodes the settings specific to the model used, aka hyperparameters. Example hyperparameters are provided in the example config.yml. Only the settings for the model selected in the Job section will be used. Model settings which can be changed in the command line are: --epochs, --batch_size, and --lr.
	
### Hyperparameter optimization

This example provides instructions for hyperparameter optimization. 

1. Similar to regular training, ensure the dataset is available with requisite files in the directory.

2. To run hyperparameter optimization, one must first define the hyperparameter search space. MatDeepLearn uses [RayTune](https://docs.ray.io/en/master/tune/index.html) for distributed optimization, and the search space is defined with their provided methods. The choice of search space will depend on many factors, including available computational resources and focus of the study; we provide some examples for the existing models in main.py.

3. Assuming the search space is defined, we run hyperparameter optimization with :
	```bash
	python main.py --data_path='your_path_here' --model='DOS_STO' --job_name="my_hyperparameter_job" --run_mode='Hyperparameter'
	```		
	this sets the run mode to hyperparameter optimization, with a set number of trials and concurrency. Concurrently sets the number of trials to be performed in parallel; this number should be higher than the number of available devices to avoid bottlenecking. The program should automatically detect number of GPUs and run on each device accordingly. Finally, an output will be written called "optimized_hyperparameters.json" which contains the hyperparameters for the model with the lowest test error. Raw results are saved in a directory called "ray_results."

## License

Distributed under the MIT License. 


## Acknowledgements

Contributors: Victor Fung, P. Ganesh, Bobby Sumpter

## Contact

Code is maintained by:

[Victor Fung](https://www.ornl.gov/staff-profile/victor-fung), fungv (at) ornl.gov

