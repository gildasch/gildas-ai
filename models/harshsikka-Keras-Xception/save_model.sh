#!/usr/bin/env sh

docker run -ti -v ${PWD}:/xception -w /xception -u $(id -u):$(id -g) tensorflow/tensorflow python xception.py
