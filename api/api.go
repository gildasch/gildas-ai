package api

import (
	"fmt"
	goimage "image"
	"net/http"
	"strings"
	"time"

	"github.com/gildasch/gildas-ai/image"
	"github.com/gildasch/gildas-ai/tensor"
	"github.com/gin-gonic/gin"
)

type Classifier interface {
	Inception(img goimage.Image) (*tensor.Predictions, error)
}

type classifierResult struct {
	Classifier  string              `json:"classifier"`
	Predictions []tensor.Prediction `json:"predictions,omitempty"`
	Timing      string              `json:"timing"`
	Error       error               `json:"error"`
}

func ClassifyHandler(classifiers map[string]Classifier) gin.HandlerFunc {
	return func(c *gin.Context) {
		imageURL := strings.TrimPrefix(c.Param("imageurl"), "/")

		img, err := image.FromURL(imageURL)
		if err != nil {
			c.AbortWithStatusJSON(
				http.StatusBadRequest,
				fmt.Sprintf("cannot read remote image %q: %v\n", imageURL, err))
			return
		}

		resp := []classifierResult{}

		for name, c := range classifiers {
			start := time.Now()
			preds, err := c.Inception(img)
			if err != nil {
				resp = append(resp, classifierResult{
					Classifier: name,
					Error:      err,
				})
				continue
			}
			elapsed := time.Since(start)

			resp = append(resp, classifierResult{
				Classifier:  name,
				Predictions: preds.Best(10),
				Timing:      elapsed.String(),
			})
		}

		c.JSON(http.StatusOK, resp)
	}
}
