FROM gildasch/tensorflow-go:v1.12-imagemagick

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
    wget --quiet https://github.com/GildasCh/gildas-ai/releases/download/v1.0/mask_rcnn_coco_tf_1.12.0.zip && \
    unzip -x mask_rcnn_coco_tf_1.12.0.zip

ENV MODELS_ROOT /models/
