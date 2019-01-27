#!/usr/bin/env sh

set -x

echo "This tutorial shows all the step for running the model at https://modeldepot.io/jbrandowski/nasnet-mobile in Go"

if [ ! -d myModel ]; then

    if [ ! -f 09a9e3fd-ebf0-46d4-bd5d-8be69d80cf44_NASNet-mobile.h5 ]; then
        echo "Downloading model from modeldepot.io"
        curl -L -o 09a9e3fd-ebf0-46d4-bd5d-8be69d80cf44_NASNet-mobile.h5 \
             http://modeldepot.io/assets/uploads/models/models/09a9e3fd-ebf0-46d4-bd5d-8be69d80cf44_NASNet-mobile.h5
    fi

    pip install keras
    python make_model.py
fi

go run main.go
