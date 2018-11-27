FROM gildasch/tensorflow-go

RUN go get github.com/gin-gonic/gin && \
    go get github.com/nfnt/resize && \
    go get github.com/pkg/errors && \
    go get github.com/satori/go.uuid && \
    go get github.com/rwcarlsen/goexif/exif && \
    go get github.com/disintegration/imaging

COPY . /go/src/github.com/gildasch/gildas-ai

WORKDIR /go/src/github.com/gildasch/gildas-ai
ENTRYPOINT ["go", "run", "main.go"]
CMD ["web"]
