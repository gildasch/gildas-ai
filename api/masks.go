package api

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"net/http"
	"strings"
	"time"

	gildasai "github.com/gildasch/gildas-ai"
	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/gin-gonic/gin"
)

type Detector interface {
	Detect(img image.Image) ([]gildasai.Mask, error)
}

type MaskResult struct {
	jpgData []byte
	elapsed time.Duration
}

func MaskHandler(detector Detector, store map[string]MaskResult) gin.HandlerFunc {
	return func(c *gin.Context) {
		imageURL := strings.TrimPrefix(c.Query("imageurl"), "/")

		if imageURL == "" {
			c.HTML(http.StatusOK, "masks.html", nil)
			return
		}

		if _, ok := store[imageURL]; !ok {
			res, err := calculateMask(detector, imageURL)
			if err != nil {
				c.AbortWithStatusJSON(
					http.StatusBadRequest,
					fmt.Sprintf("error processing image %q: %v\n", imageURL, err))
				return
			}
			store[imageURL] = *res
		}

		c.HTML(http.StatusOK, "masks.html", gin.H{
			"imageURL":     imageURL,
			"maskImageURL": fmt.Sprintf("/masks/result.jpg?imageurl=%s", imageURL),
			"elapsed":      store[imageURL].elapsed,
		})
	}
}

func MaskImageHandler(store map[string]MaskResult) gin.HandlerFunc {
	return func(c *gin.Context) {
		imageURL := strings.TrimPrefix(c.Query("imageurl"), "/")
		res, ok := store[imageURL]
		if !ok {
			c.AbortWithStatusJSON(
				http.StatusNotFound,
				fmt.Sprintf("image %q not found\n", imageURL))
			return
		}

		c.Data(http.StatusOK, "image.jpeg", res.jpgData)
	}
}

func calculateMask(detector Detector, imageURL string) (*MaskResult, error) {
	img, err := imageutils.FromURL(imageURL)
	if err != nil {
		return nil, err
	}

	start := time.Now()
	masks, err := detector.Detect(img)
	if err != nil {
		return nil, err
	}
	elapsed := time.Since(start)

	withMasks := gildasai.DrawMasks(img, masks)

	var withMasksBuf bytes.Buffer
	err = jpeg.Encode(&withMasksBuf, withMasks, nil)
	if err != nil {
		return nil, err
	}

	return &MaskResult{
		jpgData: withMasksBuf.Bytes(),
		elapsed: elapsed}, nil
}
