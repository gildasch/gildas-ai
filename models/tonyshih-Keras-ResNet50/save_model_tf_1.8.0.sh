#!/usr/bin/env sh

docker run -ti \
       -v ${PWD}:/resnet \
       -v ${PWD}/../harshsikka-Keras-Xception/gorge.jpg:/resnet/gorge.jpg \
       -w /resnet \
       -u $(id -u):$(id -g) \
       gildasch/tensorflow-keras:v1.8.0 \
       python resnet.py resnet_tf_1.8.0
