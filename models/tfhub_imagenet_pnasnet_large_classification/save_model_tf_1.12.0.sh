#!/usr/bin/env sh

mkdir -p module_cache

docker run -ti \
       -v ${PWD}:/pnasnet \
       -v ${PWD}/../harshsikka-Keras-Xception/gorge.jpg:/pnasnet/gorge.jpg \
       -v ${PWD}/module_cache:/module_cache \
       -e TFHUB_CACHE_DIR=/module_cache \
       -w /pnasnet \
       -u $(id -u):$(id -g) \
       gildasch/tensorflow-keras \
       python pnasnet.py pnasnet_tf_1.12.0
