#!/usr/bin/env sh

docker run \
       -v ${PWD}/main.go:/running-tensorflow-model/main.go \
       -v ${PWD}/run_tutorial.sh:/running-tensorflow-model/run_tutorial.sh \
       -w /running-tensorflow-model \
       gildasch/tensorflow-go \
       ./run_tutorial.sh
