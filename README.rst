TORCHIVA
========

A package for blind source separation and beamforming in `pytorch <https://pytorch.org>`_ .

* supports many BSS and beamforming methods
* supports memory efficient gradient computation for training neural source models
* supports batched computations
* can run on GPU via pytorch

Author
------

* `Robin Scheibler <robin.scheibler@linecorp.com>`_
* Kohei Saijo


Quick Start
-----------

This guide assumes `anaconda <https://www.anaconda.com/products/individual>`_ is installed::

    # get code and install environment
    git clone <torchiva_repo>
    cd torchiva
    conda env create -f environment.yml
    conda activate torchiva
    pip install -e .

    cd ./examples
    export PYTHONPATH="/path/to/torchiva":$PYTHONPATH"

    # BSS example
    # algorithm can be selected from tiss, auxiva_ip, auxiva_ip2, and five
    python ./example.py PATH_TO_DATASET ALGORITHM


Separation using Pre-trained Model
----------------------------------

We provide pre-trained model at --- hugging face link ---.
The model is trained with `WSJ1-mix dataset <https://github.com/fakufaku/create_wsj1_2345_db>`_ with the same configuration as `./configs/tiss.json`.
You can easily try separation with the pre-trained model::

    # download model parameters from hugging face

    # Separation
    python ./example_dnn.py ./configs/tiss.json PATH_TO_DATASET PATH_TO_MODEL_PARAMS


Training
--------

We provide some simple training scripts.
We support training of **T-ISS**, **MWF**, **MVDR**, **GEV**::

    cd examples

    # install some modules necessary for training
    pip install -r requirements.txt

    # training
    python train.py PATH_TO_CONFIG PATH_TO_DATASET


Note that our example scripts assumes using WSJ1-mix dataset.
If you want to use other datasets, please change the script in the part that loads audios.

Test your trained model with checkpoint from epoch 128::

    # python ./test.py --dataset ../wsj1_6ch --n_fft 2048 --hop 512 --n_iter 40 --iss-hparams checkpoints/tiss_delay1tap5_2ch/lightning_logs/version_0/hparams.yaml --epoch 128 --test

Export the trained model for later use::

    python ./export_model.py ../trained_models/tiss checkpoints/tiss_delay1tap5_2ch/lightning_logs/version_0 128 146 148 138 122 116 112 108 104 97

Run the example script using the exported model::

    python ./example_dnn.py ../wsj1_6ch ../trained_models/tiss -m 2 -r 100

License
-------

2022 (c) Robin Scheibler, Kohei Saijo, LINE Corporation.

All of this code is released under `MIT License <https://opensource.org/licenses/MIT>`_
