FROM gildasch/tensorflow-go

RUN go get github.com/gin-gonic/gin && \
    go get github.com/nfnt/resize && \
    go get github.com/pkg/errors

COPY . /go/src/github.com/gildasch/gildas-ai

WORKDIR /go/src/github.com/gildasch/gildas-ai
ENTRYPOINT ["go", "run", "main.go"]
CMD ["web"]
