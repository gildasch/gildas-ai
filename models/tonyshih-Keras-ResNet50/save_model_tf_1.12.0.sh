#!/usr/bin/env sh

docker run -ti \
       -v ${PWD}:/resnet \
       -v ${PWD}/../harshsikka-Keras-Xception/gorge.jpg:/resnet/gorge.jpg \
       -w /resnet \
       -u $(id -u):$(id -g) \
       gildasch/tensorflow-keras \
       python resnet.py resnet_tf_1.12.0
