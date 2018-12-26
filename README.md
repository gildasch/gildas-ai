# gildas-ai [![Build Status](https://travis-ci.org/GildasCh/gildas-ai.svg?branch=master)](https://travis-ci.org/GildasCh/gildas-ai)

Easy access to AI tasks (starting with object detection) as a web
interface, a JSON API and the command line.

![wine](static/wine.png)

![faces](static/faces_wide.png)

![faceswap](static/faceswap.jpg)

![masks](static/masks.jpg)

With Tensorflow and the Go bindings installed, run it with:

```
go run main.go web
```

Using Docker:

```
docker run -p 8080:8080 gildasch/gildas-ai
```

Models used for object detection on ImageNet:

- Keras Xception from https://modeldepot.io/harshsikka/keras-xception
- Keras ResNet50 from https://modeldepot.io/tonyshih/keras-resnet50
- Keras NASNet Mobile from https://modeldepot.io/jbrandowski/nasnet-mobile
- Tensorflow Hub PNASNet-5 (large) from https://tfhub.dev/google/imagenet/pnasnet_large/classification/2

Models used for face detection and recognition:

- face-api.js from https://itnext.io/face-api-js-javascript-api-for-face-recognition-in-the-browser-with-tensorflow-js-bcc2a6c4cf07 / https://github.com/justadudewhohacks/face-api.js

Model used for object detection and segmentation on COCO:

- Keras Mask R-CNN from https://modeldepot.io/dani/mask-r-cnn
