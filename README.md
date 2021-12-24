# Final Project

This repository contains the code and various resources used to complete our
final project for LING 380. Please note that the repositiory contains some large
files (containing training data), and as such, `git lfs` must be installed
(see (https://git-lfs.github.com/)[https://git-lfs.github.com/]).

The `paper` directory contains the various `tex` files and images used to
generate the final writeup.

The `src` directory contains the code used tot train and analyse the model. To
train a new model, run `python train.py`, and for more information on how to
run, apply the `-h` command line option. The file `main.py` contains various
utility functions that were used in anlysing data; it does not do anything when
run in isolation. Dependencies include: `pytorch`, `pickle`, `pandas`,
`matplotlib`, `seaborn`, `scipy`, and `pingouin`.

The `src/data` directory contains the datasets used for training the model.

The `experiment_data` directory contains the test items used for experiments
outlined in the paper.


