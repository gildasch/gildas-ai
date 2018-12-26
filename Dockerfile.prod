FROM gildasch/tensorflow-go:v1.8-imagemagick

RUN go get github.com/gin-gonic/gin && \
    go get github.com/nfnt/resize && \
    go get github.com/pkg/errors && \
    go get github.com/satori/go.uuid && \
    go get github.com/rwcarlsen/goexif/exif && \
    go get github.com/disintegration/imaging && \
    go get github.com/mattn/go-sqlite3 && \
    go get gopkg.in/gographics/imagick.v3/imagick && \
    go get github.com/fogleman/gg && \
    go get github.com/lucasb-eyer/go-colorful && \
    go get github.com/esimov/colorquant && \
    go get github.com/gin-contrib/cache && \
    go get github.com/disintegration/imaging && \
    go get github.com/stretchr/testify/assert && \
    go get github.com/stretchr/testify/require

RUN mkdir /models && cd /models && \
    wget --quiet https://github.com/GildasCh/gildas-ai/releases/download/v1.0/face-api-js-landmarks-face-api-landmarksnet_tf_1.8.0.zip && \
    wget --quiet https://github.com/GildasCh/gildas-ai/releases/download/v1.0/face-api-js-descriptors-face-api-descriptors_tf_1.8.0.zip && \
    wget --quiet https://github.com/GildasCh/gildas-ai/releases/download/v1.0/harshsikka-Keras-Xception-xception_tf_1.8.0.zip && \
    wget --quiet https://github.com/GildasCh/gildas-ai/releases/download/v1.0/tonyshih-Keras-ResNet50-resnet_tf_1.8.0.zip && \
    wget --quiet https://github.com/GildasCh/gildas-ai/releases/download/v1.0/jbrandowski_NASNet_Mobile-nasnet-mobile_tf_1.8.0.zip && \
    wget --quiet https://github.com/GildasCh/gildas-ai/releases/download/v1.0/tfhub_imagenet_pnasnet_large_classification-pnasnet_tf_1.8.0.zip && \
    wget --quiet https://github.com/GildasCh/gildas-ai/releases/download/v1.0/mask_rcnn_coco_tf_1.8.0.zip && \
    unzip -x face-api-js-landmarks-face-api-landmarksnet_tf_1.8.0.zip && \
    unzip -x face-api-js-descriptors-face-api-descriptors_tf_1.8.0.zip && \
    unzip -x harshsikka-Keras-Xception-xception_tf_1.8.0.zip && \
    unzip -x tonyshih-Keras-ResNet50-resnet_tf_1.8.0.zip && \
    unzip -x jbrandowski_NASNet_Mobile-nasnet-mobile_tf_1.8.0.zip && \
    unzip -x tfhub_imagenet_pnasnet_large_classification-pnasnet_tf_1.8.0.zip && \
    unzip -x mask_rcnn_coco_tf_1.8.0.zip && \
    rm *.zip

ENV MODELS_ROOT /models/

COPY . /go/src/github.com/gildasch/gildas-ai
WORKDIR /go/src/github.com/gildasch/gildas-ai

ENTRYPOINT ["go", "run", "cmd/gildas-ai/main.go"]
CMD ["web"]
