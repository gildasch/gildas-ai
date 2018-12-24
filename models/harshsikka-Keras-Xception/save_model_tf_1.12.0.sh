#!/usr/bin/env sh

docker run -ti \
       -v ${PWD}:/xception \
       -w /xception \
       -u $(id -u):$(id -g) \
       gildasch/tensorflow-keras \
       python xception.py xception_tf_1.12.0
