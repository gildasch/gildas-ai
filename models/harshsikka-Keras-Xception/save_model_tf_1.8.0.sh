#!/usr/bin/env sh

docker run -ti \
       -v ${PWD}:/xception \
       -w /xception \
       -u $(id -u):$(id -g) \
       gildasch/tensorflow-keras:v1.8.0 \
       python xception.py xception_tf_1.8.0
