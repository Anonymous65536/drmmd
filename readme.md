# (De)-regularized Maximum Mean Discrepancy Gradient Flow

This repository contains the implementation of the code for the paper "(De)-regularized Maximum Mean Discrepancy Gradient Flow". 
## Installation

To install the necessary requirements, use the following command:

`pip install -r requirements.txt`

## Reproducing Results

### Three-ring experiment

To reproduce the results for Three-ring experiment, run the following command:

`python run.py --dataset ThreeRing --flow chard --kernel Gaussian --lmbda 0.001 --step_size 1e-5 --bandwidth 0.1 --step_num 100000 --source_particle_num 300 --opt sgd --seed 42`

You can vary the deregularization coefficient by altering the argument of `lmbda`.
`--flow chard` is DrMMD flow, `--flow mmd` is MMD flow

### Student-teacher neural network

To reproduce the results for training student-teacher neural network, run the following command:

`python student_teacher/train.py --device 0 --lmbda 0.1 --loss chard --lr 0.1 --with_noise false --noise_decay_freq 500 --seed 42 --log_in_file`

You can vary the deregularization coefficient by altering the argument of `lmbda`.
`--loss chard` is DrMMD flow, and `--with_noise false` controls whether to use noise injection.