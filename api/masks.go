package api

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"net/http"
	"strings"
	"time"

	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/gildasch/gildas-ai/mask"
	"github.com/gin-gonic/gin"
)

type Detector interface {
	Detect(img image.Image) (*mask.Detections, *mask.Masks, error)
}

func MaskHandler(detector Detector, store map[string][]byte) gin.HandlerFunc {
	return func(c *gin.Context) {
		imageURL := strings.TrimPrefix(c.Query("imageurl"), "/")

		if imageURL == "" {
			c.HTML(http.StatusOK, "masks.html", nil)
			return
		}

		img, err := imageutils.FromURL(imageURL)
		if err != nil {
			c.AbortWithStatusJSON(
				http.StatusBadRequest,
				fmt.Sprintf("cannot read remote image %q: %v\n", imageURL, err))
			return
		}

		start := time.Now()
		detecs, masks, err := detector.Detect(img)
		if err != nil {
			c.AbortWithStatusJSON(
				http.StatusBadRequest,
				fmt.Sprintf("cannot detect from remote image %q: %v\n", imageURL, err))
			return
		}
		elapsed := time.Since(start)

		withMasks := masks.DrawMasks(detecs, img)

		var withMasksBuf bytes.Buffer
		err = jpeg.Encode(&withMasksBuf, withMasks, nil)
		if err != nil {
			c.AbortWithStatusJSON(
				http.StatusInternalServerError,
				fmt.Sprintf("could not encode result: %v\n", err))
			return
		}

		store[imageURL] = withMasksBuf.Bytes()

		c.HTML(http.StatusOK, "masks.html", gin.H{
			"imageURL":     imageURL,
			"maskImageURL": fmt.Sprintf("/masks/result.jpg?imageurl=%s", imageURL),
			"elapsed":      elapsed,
		})
	}
}

func MaskImageHandler(store map[string][]byte) gin.HandlerFunc {
	return func(c *gin.Context) {
		imageURL := strings.TrimPrefix(c.Query("imageurl"), "/")
		data, ok := store[imageURL]
		if !ok {
			c.AbortWithStatusJSON(
				http.StatusNotFound,
				fmt.Sprintf("image %q not found\n", imageURL))
			return
		}

		c.Data(http.StatusOK, "image.jpeg", data)
	}
}
