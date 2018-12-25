#!/usr/bin/env sh

docker run -ti \
       -v ${PWD}:/face-api-descriptors \
       -v ${PWD}/../harshsikka-Keras-Xception/gorge.jpg:/face-api-descriptors/gorge.jpg \
       -w /face-api-descriptors \
       -u $(id -u):$(id -g) \
       gildasch/tensorflow-keras:v1.8.0 \
       python descriptors_tf_1.8.0.py face-api-descriptors_tf_1.8.0
