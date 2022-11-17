Samples
-------

This folder contains some samples for tests and examples.

You can create your own multichannel mixtures as follows.

1. copy some dry sources into the `dry` folder
2. add the file names of the dry sources and the mixture and references to create
  in `samples_list.yaml` following the same pattern as the sample already present
3. Install dependencies
    ```
    pip install numpy, scipy, PyYAML, pyroomacoustics
    ```
4. Run
    ```shell
    python ./make_mix.py
    ```
