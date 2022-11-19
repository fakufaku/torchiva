#!/bin/bash

INPUT=./examples/samples/mix_reverb
OUTPUT=./test_sep
DEVICE=cpu

# test with all default values
python -m torchiva.separate ${INPUT} ${OUTPUT}/default

for mic in 2 3; do
    python -m torchiva.separate ${INPUT} ${OUTPUT}/tiss-nn-m${mic} \
        -a tiss -m ${mic} -s 2 --model-type nn --device ${DEVICE}
done

# test classic models
for model in laplace gauss nmf; do
    python -m torchiva.separate ${INPUT} ${OUTPUT}/ip2-${model} \
        -a ip2 -m 2 -s 2 --model-type ${model} --device ${DEVICE}

    for mic in 2 3; do
        python -m torchiva.separate ${INPUT} ${OUTPUT}/tiss-${model}-m${mic} \
            -a tiss -m ${mic} -s 2 --model-type ${model} --device ${DEVICE}
        python -m torchiva.separate ${INPUT} ${OUTPUT}/tiss-${model}-m${mic}-t2 \
            -a tiss -m ${mic} -s 2 -d 3 -t 2 --model-type ${model} --device ${DEVICE}

        python -m torchiva.separate ${INPUT} ${OUTPUT}/five-${model}-m${mic} \
            -a five -m ${mic} -s 1 --model-type ${model} --device ${DEVICE}
    done
done
