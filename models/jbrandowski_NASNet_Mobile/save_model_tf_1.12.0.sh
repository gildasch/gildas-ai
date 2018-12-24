#!/usr/bin/env sh

docker run -ti \
       -v ${PWD}:/nasnet-mobile \
       -v ${PWD}/../harshsikka-Keras-Xception/gorge.jpg:/nasnet-mobile/gorge.jpg \
       -w /nasnet-mobile \
       -u $(id -u):$(id -g) \
       gildasch/tensorflow-keras \
       python nasnet-mobile.py nasnet-mobile_tf_1.12.0
