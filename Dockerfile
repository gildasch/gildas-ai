FROM gildasch/tensorflow-go:v1.8

RUN apt-get -y update

RUN apt-get -y install libtcmalloc-minimal4
RUN export LD_PRELOAD="/usr/lib/libtcmalloc_minimal.so.4"

RUN apt-get -y install build-essential checkinstall libx11-dev libxext-dev zlib1g-dev libpng12-dev libjpeg-dev libfreetype6-dev libxml2-dev wget && \
    cd /tmp && wget http://www.imagemagick.org/download/ImageMagick-7.0.8-19.tar.gz && \
    tar xvzf ImageMagick-7.0.8-19.tar.gz && cd ImageMagick-7.0.8-19 && \
    touch configure && ./configure && make && make install && \
    ldconfig /usr/local/lib && \
    rm -rf /tmp/ImageMagick*

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
    go get github.com/disintegration/imaging

COPY . /go/src/github.com/gildasch/gildas-ai

WORKDIR /go/src/github.com/gildasch/gildas-ai
ENTRYPOINT ["go", "run", "main.go"]
CMD ["web"]
