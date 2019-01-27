#!/usr/bin/env sh

docker run \
       -v ${PWD}/make_model.py:/image-recognition-with-jbrandowskis-nasnet-mobile/make_model.py \
       -v ${PWD}/main.go:/image-recognition-with-jbrandowskis-nasnet-mobile/main.go \
       -v ${PWD}/run_tutorial.sh:/image-recognition-with-jbrandowskis-nasnet-mobile/run_tutorial.sh \
       -v ${PWD}/cat.jpg:/image-recognition-with-jbrandowskis-nasnet-mobile/cat.jpg \
       -w /image-recognition-with-jbrandowskis-nasnet-mobile \
       gildasch/tensorflow-go \
       ./run_tutorial.sh
