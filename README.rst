TORCHIVA
========

A package for blind source separation and beamforming in `pytorch <https://pytorch.org>`_ .

* supports many BSS and beamforming methods
* supports memory efficient gradient computation for training neural source models
* supports batched computations
* can run on GPU via pytorch

Quick Start
-----------

The package can be installed via pip::

    pip install torchiva

Separation using Pre-trained Model
----------------------------------

We provide a pre-trained model in `trained_models/tiss`.
You can easily try separation with the pre-trained model::

    # Separation
    python -m torchiva.separation INPUT OUTPUT

where ``INPUT`` is either a multichannel wav file or a folder containing
multichannel wav files.  If a folder, then all the files inside are separted.
The output is saved to ``OUTPUT``.
The model stored in ``trained_models/tiss`` is automatically downloaded to
``$HOME/.torchiva_models``. The path or url to the model can also be
manually provided via the ``--model`` option.
The model was trained on the `WSJ1-mix dataset
<https://github.com/fakufaku/create_wsj1_2345_db>`_ with the same configuration
as ``./examples/configs/tiss.json``.


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

Authors
-------

* `Robin Scheibler <robin.scheibler@linecorp.com>`_
* Kohei Saijo

License
-------

2022 (c) Robin Scheibler, Kohei Saijo, LINE Corporation.

All of this code is released under `MIT License <https://opensource.org/licenses/MIT>`_
