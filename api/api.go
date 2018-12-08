package api

import (
	"fmt"
	"image"
	"net/http"
	"strings"
	"time"

	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/gildasch/gildas-ai/objects/classifiers"
	"github.com/gin-gonic/gin"
)

type Classifier interface {
	Inception(img image.Image) (*classifiers.Predictions, error)
}

type classifierResult struct {
	Classifier  string                   `json:"classifier"`
	Predictions []classifiers.Prediction `json:"predictions,omitempty"`
	Timing      string                   `json:"timing"`
	Error       error                    `json:"error"`
}

func ClassifyHandler(classifiers map[string]Classifier, html bool) gin.HandlerFunc {
	return func(c *gin.Context) {
		imageURL := strings.TrimPrefix(c.Query("imageurl"), "/")

		if imageURL == "" {
			c.HTML(http.StatusOK, "predictions.html", nil)
			return
		}

		img, err := imageutils.FromURL(imageURL)
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

		if html {
			c.HTML(http.StatusOK, "predictions.html", gin.H{
				"imageURL": imageURL,
				"results":  resp,
			})
			return
		}

		c.JSON(http.StatusOK, resp)
	}
}
