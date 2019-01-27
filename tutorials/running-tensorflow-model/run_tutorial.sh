#!/usr/bin/env sh

set -x

if [ ! -d myModel ]; then
    if [ ! -f jbrandowski_NASNet_Mobile-nasnet-mobile_tf_1.12.0.zip ]; then
        curl -L -o jbrandowski_NASNet_Mobile-nasnet-mobile_tf_1.12.0.zip \
             https://github.com/gildasch/gildas-ai/releases/download/v1.0/jbrandowski_NASNet_Mobile-nasnet-mobile_tf_1.12.0.zip
    fi
    unzip -x jbrandowski_NASNet_Mobile-nasnet-mobile_tf_1.12.0.zip
    mv models/jbrandowski_NASNet_Mobile/nasnet-mobile_tf_1.12.0 myModel
    rmdir -p models/jbrandowski_NASNet_Mobile
fi

go run main.go
