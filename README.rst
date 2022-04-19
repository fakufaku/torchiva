TORCHIVA
========

A package for blind source separation and beamforming in `pytorch <https://pytorch.org>`_ .

* supports many BSS methods
* supports memory efficient gradient computation for training neural source models
* supports batched computations
* can run on GPU via pytorch

Author
------

* `Robin Scheibler <robin.scheibler@linecorp.com>`_
* Kohei Saijo


Quick Start
-----------

This supposes `anaconda <https://www.anaconda.com/products/individual>`_ is installed::

    # get code and install environment
    git@git.linecorp.com:speechresearch/torchiva.git
    cd torchiva
    conda env create -f environment.yml
    conda activate torchiva

    # BSS example
    python ./example.py DATA_PATH ALGORITHM


License
-------

2022 (c) Robin Scheibler, LINE Corporation
All of this code is released under `MIT License <https://opensource.org/licenses/MIT>`_




