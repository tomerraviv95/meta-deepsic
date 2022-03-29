*"A baby has brains, but it doesn’t know much. Experience is the only thing that brings knowledge, and the longer you are on earth the more experience you are sure to get."* 

--Wizard of Oz.

# Meta-DeepSIC repository

Python repository for the paper "Online Meta-Learning For Hybrid Model-Based Deep Receivers".

Please cite our [paper](https://arxiv.org/abs/2203.14359), if the code is used for publishing research.

# Table of Contents

- [Introduction](#introduction)
- [Folders Structure](#folders-structure)
  * [python_code](#python_code)
    + [data](#data)
    + [detectors](#detectors)
    + [ecc](#ecc)
    + [plotting](#plotting)
    + [trainers](#trainers)
    + [utils](#utils)
  * [resources](#resources)
  * [dir_definitions](#dir_definitions)
- [Execution](#execution)
  * [Environment Installation](#environment-installation)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

# Introduction

This repository implements classical and machine-learning based detectors for a channel with memory of L in the MIMO case. Now, the brute force approach is obviously infeasible as the number of users and antenna increases - requiring a suboptimal yet more time-efficient approach. Soft Interference Cancellation (SIC) is an iterative method, working by calculating posteriors per user, and combining them iteratively. In the [DeepSIC method](https://arxiv.org/abs/2002.03214), the model-based building blocks of iterative SIC are replaced with simple dedicated DNNs, thus being able to cope better with non-linear channels or in cases of CSI uncertainty. This repository builds on the DeepSIC architecture and improves the training algorithm. The training method we propose is tailored for dynamic communications scenarios, where the channels are changing in time. To that end, we adopt a meta-learning algorithm for the time-series detection problem. The codebase used is explained below.

# Folders Structure

## python_code 

The python simulations of the simplified communication chain: encoder, channel and detectors.

### data 

Includes the channel dataloader (in the data_generator module) and the channel model, including the SED and beamformed COST channels. In particular, the dataloader generates pairs of (transmitted,received) samples. 

### detectors

The backbone detectors: DeepSIC, Meta-DeepSIC, BlackBox, Meta-BlackBox. The meta and non-meta detectors have slightly different API so they are seperated in the trainer class below. The meta variation has to receive the parameters of the original architecture for the training. Each DeepSIC network employs 3 linear layers, with sigmoid and relu activations inbetween. The black-box architectures employs Resnet blocks from [this post](https://stackoverflow.com/questions/60817390/implementing-a-simple-resnet-block-with-pytorch).

### ecc

Error-correction codes functions. Code from [site](https://en.wikiversity.org/wiki/Reed%E2%80%93Solomon_codes_for_coders).

### plotting

Features main plotting tools for the paper:

* basic_plotter - features the main plotting class, with plots of coded BER by block index and SNR, or SER by pilots length.
* plotter_config - colors, markers and linestyles for all methods.
* plotter_utils - wrapper for the trainers initialization and method name.
* plot_figures_for_paper - loads the relevant config from config_runs directory, and runs the appropriate methods for plotting the MIMO figures in the paper.

### trainers 

Wrappers for the training and evaluation of the detectors. Trainer holds the training, sequential evaluation of data blocks with coding and without pilots / with pilots and without coding. It also holds the main function that trains the detector and evaluates it, returning a list of coded ber/ser per block. The train and test dataloaders are also initialized by the trainer.

The specific class trainer (deepsic/blackbox) holds relevant functions that are used by that specific trainer only. In each directory we also have a wrapper trainer for each detector, that implements different training methodologies. Joint implements training in offline only, not online. Online methods implement both the training and online training as the same util. Meta methods implement both the offline and online meta-training, as well as the online supervised training. 

Each trainer is executable by running it. The trainer runs the main function of trainer - training and evaluation one after the other.

### utils

Extra utils for pickle manipulations and tensor reshaping; calculating the accuracy over FER and BER; several constants; and most important - the config singleton class.

The config works by the [singleton design pattern](https://en.wikipedia.org/wiki/Singleton_pattern). Check the link if unfamiliar. 

The config is accessible from every module in the package, featuring the next parameters:

1. n_user - number of transmitting users. Integer.
2. n_ant - number of receiving antennas. Integer.
3. snr - channel signal-to-noise ratio, in dB. Integer.
4. iterations - number of iterations in the unfolded DeepSIC architecture. Integer.
5. info_size - number of information bits in each training pilot block and test data block. Integer.
6. train_frame_num - number of blocks used for training. Integer.
7. test_frame_num - number of blocks used for test. Integer.
8. test_pilot_size - number of bits in each test pilot block. Integer.
9. fading - whether to use fading. Relevant only to the SED channel. Boolean flag.
10. channel_mode - choose the Spatial Exponential Decay Channel Model, i.e. exp(-|i-j|), or the beamformed COST channel. String value: 'SED' or 'COST'. COST works with 8x8 n_user and n_ant only.
11. lr - learning rate for training. Float.
12. max_epochs - number of offline training epochs. Integer.
13. self_supervised_epochs - number of online training epochs. Integer.
14. use_ecc - whether to use Error Correction Codes (ECC) or not. If not - automatically will run in evaluations the online pilots-blocks scenario (as in pilots efficiency part). Boolean flag.
15. n_ecc_symbols - number of symbols in ecc. Number of additional transmitted bits is 8 times this value, due to the specific Reed-Solomon we employ. [Read more here](https://en.wikiversity.org/wiki/Reed%E2%80%93Solomon_codes_for_coders). Integer.
16. ber_thresh - threshold for self-supervised training, as in the [ViterbiNet](https://arxiv.org/abs/1905.10750) or [DeepSIC](https://arxiv.org/abs/2002.03214) papers.
17. change_user_only - allows change of channel for a single user. Integer value (the index of the desired user: {0,..,n_user}).
18. retrain_user - only in the DeepSIC architecture, allow for training the specific user networks only. Integer value (the index of the desired user: {0,..,n_user}).

## resources

Keeps the COST channel coefficients vectors in 4 test folders. Also holds config runs for the paper.

## dir_definitions 

Definitions of relative directories.

# Execution

To execute the code, first download and install Git, Anaconda and PyCharm.

Then install the environment, follow the installation setup below. 

At last, open PyCharm in the root directory. You may run either the trainers or one of the plotters.

This code was simulated with GeForce RTX 2060 with driver version 432.00 and CUDA 10.1. 

## Environment Installation

1. Open git bash and cd to a working directory of you choice.

2. Clone this repository to your local machine.

3. Open Anaconda prompt and navigate to the cloned repository.

4. Run the command "conda env create -f metadeepsic.yml". This should install the required python environment.

5. Open the cloned directory using PyCharm.

6. After the project has been opened in PyCharm, go to settings, File -> Settings... (or CTRL ALT S)

7. In the opened window open the tab Project -> Project Interpreter

8. In the new window, click on the cog icon and then on Add...

9. In the add python interpreter window, click on the Conda Environment tab

10. Select Existing environment and navigate to where the python.exe executable of the deep_ensemble environment is installed under the interpreter setting

  - For windows its usually found at C:\users\<username>\anaconda3\envs\metadeepsic\python.exe)

  - For linux its usually found at /home/<username>/anaconda3
  
11. Click OK

12. Done!
