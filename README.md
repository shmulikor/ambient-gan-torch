# AmbientGAN-torch: Generative models from lossy measurements in pytorch #

The code in this repository, developed in Python3 and PyTorch, is designed for reproducing the results presented in the paper [AmbientGAN: Generative models from lossy measurements](https://openreview.net/forum?id=Hy7fDog0b). It is adapted from [this](https://github.com/AshishBora/ambient-gan) repository, which was originally implemented in Python 2.7 and TensorFlow. Additionally, the code was modified to accommodate 3D images.

The training setup is as in the following diagram:

<img src="https://github.com/AshishBora/ambient-gan/blob/master/setup.png" width="350" height="200">

Here are a few example results:

The left image is a sample of measured MNIST data and the right shows a sample of the model output.
<img src="https://github.com/shmulikor/ambient-gan-torch/blob/master/images/mnist_drop09_measurements.png" width="200" height="200"><img src="https://github.com/shmulikor/ambient-gan-torch/blob/master/images/mnist_drop09_model_out.png" width="200" height="200">

<img src="https://github.com/shmulikor/ambient-gan-torch/blob/master/images/celebA_keep_measurements.png" width="200" height="200"><img src="https://github.com/shmulikor/ambient-gan-torch/blob/master/images/celebA_keep_model_out.png" width="200" height="200">

## Requirements ##

For `pip` installation, use `$ pip install -r requirements.txt`

## Get the data ##

- MNIST data is automatically downloaded
- Get the celebA dataset [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and put the jpeg files in ./data/celebA/

## Create experiment scripts ##

Run `./create_scripts/create_scripts.sh`

This will create scripts for all the experiments in the paper.

[Optional] If you want to run only a subset of experiments you can define the grid in `./create_scripts/DATASET_NAME/grid_*.sh` or if you wish to tweak a lot of parameters, you can change `./create_scripts/DATASET_NAME/base_script.sh.` Then run `./create_scripts/create_scripts.sh` as above to create the corresponding scripts (remember to remove any previous files from `./scripts/`).

## Run experiments ##

We provide scripts to train on multiple GPUs in parallel. For example, if you wish to use 4 GPUs, you can run: `./run_scripts/run_sequentially_parallel.sh "0 1 2 3"`

This will start 4 GNU screens. Each program within the screen will attempt to acquire and run experiments from `./scripts/`, one at a time. Each experiment run will save samples, checkpoints, etc. to `./results/`.

## See results as you train ##

You can see samples for each experiment in `./results/samples/EXPT_DIR/`

EXPT_DIR is defined based on the hyperparameters of the experiment. See `./src/commons/dir_def.py` to see how this is done.
