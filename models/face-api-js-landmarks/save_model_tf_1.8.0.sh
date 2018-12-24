#!/usr/bin/env sh

mkdir -p module_cache

docker run -ti \
       -v ${PWD}:/face-api-landmarksnet \
       -v ${PWD}/../harshsikka-Keras-Xception/gorge.jpg:/face-api-landmarksnet/gorge.jpg \
       -v ${PWD}/module_cache:/module_cache \
       -e TFHUB_CACHE_DIR=/module_cache \
       -w /face-api-landmarksnet \
       -u $(id -u):$(id -g) \
       gildasch/tensorflow-keras:v1.8.0 \
       python landmarks_tf_1.8.0.py face-api-landmarksnet_tf_1.8.0
