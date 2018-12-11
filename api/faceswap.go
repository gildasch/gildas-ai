package api

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"html/template"
	"image"
	"image/draw"
	"image/gif"
	"image/jpeg"
	"net/http"
	"strings"

	"github.com/gildasch/gildas-ai/faces/swap"
	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/gin-gonic/gin"
)

func FaceSwapHandler(extractor swap.Extractor, detector swap.LandmarkDetector) gin.HandlerFunc {
	return func(c *gin.Context) {
		srcURL := strings.TrimPrefix(c.Query("src"), "/")
		dstURL := strings.TrimPrefix(c.Query("dst"), "/")

		if srcURL == "" || dstURL == "" {
			c.HTML(http.StatusOK, "faceswap.html", gin.H{
				"src": srcURL,
				"dst": dstURL,
			})
			return
		}

		src, err := imageutils.FromURL(srcURL)
		if err != nil {
			c.AbortWithStatusJSON(
				http.StatusBadRequest,
				fmt.Sprintf("cannot read remote image %q: %v\n", srcURL, err))
			return
		}

		if strings.HasSuffix(strings.ToLower(dstURL), ".gif") {
			dstGIF, err := imageutils.GIFFromURL(dstURL)
			if err != nil {
				c.AbortWithStatusJSON(
					http.StatusBadRequest,
					fmt.Sprintf("cannot read remote image %q: %v\n", dstURL, err))
				return
			}

			if len(dstGIF.Image) == 0 {
				c.AbortWithStatusJSON(
					http.StatusBadRequest,
					fmt.Sprintf("gif %q has zero frame\n", dstURL))
				return
			}

			dst := image.NewRGBA(dstGIF.Image[0].Bounds())
			for i := 0; i < len(dstGIF.Image); i++ {
				draw.Draw(dst, dst.Bounds(), dstGIF.Image[i], dstGIF.Image[i].Bounds().Min, draw.Over)

				out, err := swap.FaceSwap(extractor, detector, dst, src)
				if err != nil {
					continue
				}

				outPaletted := image.NewPaletted(out.Bounds(), dstGIF.Image[i].Palette)
				draw.Draw(outPaletted, outPaletted.Rect, out, out.Bounds().Min, draw.Over)

				dstGIF.Image[i] = outPaletted
			}

			c.HTML(http.StatusOK, "faceswap.html", gin.H{
				"src": srcURL,
				"dst": dstURL,
				"out": template.URL(toHTMLBase64GIF(dstGIF)),
			})
		}

		dst, err := imageutils.FromURL(dstURL)
		if err != nil {
			c.AbortWithStatusJSON(
				http.StatusBadRequest,
				fmt.Sprintf("cannot read remote image %q: %v\n", dstURL, err))
			return
		}

		out, err := swap.FaceSwap(extractor, detector, dst, src)
		if err != nil {
			c.AbortWithStatusJSON(
				http.StatusBadRequest,
				fmt.Sprintf("error faceswapping: %v\n", err))
			return
		}

		c.HTML(http.StatusOK, "faceswap.html", gin.H{
			"src": srcURL,
			"dst": dstURL,
			"out": template.URL(toHTMLBase64(out)),
		})
	}
}

func toHTMLBase64(img image.Image) string {
	var buf bytes.Buffer
	_ = jpeg.Encode(&buf, img, nil)

	b64 := base64.StdEncoding.EncodeToString(buf.Bytes())
	return "data:image/jpeg;base64," + b64
}

func toHTMLBase64GIF(g *gif.GIF) string {
	var buf bytes.Buffer
	_ = gif.EncodeAll(&buf, g)

	b64 := base64.StdEncoding.EncodeToString(buf.Bytes())
	return "data:image/gif;base64," + b64
}
