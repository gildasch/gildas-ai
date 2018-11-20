package api

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"html/template"
	goimage "image"
	"image/draw"
	"image/jpeg"
	"net/http"

	"github.com/gildasch/gildas-ai/faces/descriptors"
	"github.com/gildasch/gildas-ai/faces/detection"
	"github.com/gildasch/gildas-ai/faces/landmarks"
	"github.com/gildasch/gildas-ai/image"
	"github.com/gin-gonic/gin"
	"github.com/pkg/errors"
)

func FacesHandler(detector *detection.Detector, landmark *landmarks.Landmark,
	descriptor *descriptors.Descriptor) gin.HandlerFunc {
	return func(c *gin.Context) {
		imageURL1 := c.Query("image1")
		imageURL2 := c.Query("image2")

		if imageURL1 == "" || imageURL2 == "" {
			c.HTML(http.StatusOK, "faces.html", nil)
			return
		}

		img1, err := image.FromURL(imageURL1)
		if err != nil {
			c.AbortWithStatusJSON(
				http.StatusBadRequest,
				fmt.Sprintf("cannot read remote image %q: %v\n", imageURL1, err))
			return
		}

		cropped1, descr1, err := extract(img1, detector, landmark, descriptor)
		if err != nil {
			c.AbortWithStatusJSON(
				http.StatusBadRequest,
				fmt.Sprintf("cannot extract descriptors from image %q: %v\n", imageURL1, err))
			return
		}

		img2, err := image.FromURL(imageURL2)
		if err != nil {
			c.AbortWithStatusJSON(
				http.StatusBadRequest,
				fmt.Sprintf("cannot read remote image %q: %v\n", imageURL2, err))
			return
		}

		cropped2, descr2, err := extract(img2, detector, landmark, descriptor)
		if err != nil {
			c.AbortWithStatusJSON(
				http.StatusBadRequest,
				fmt.Sprintf("cannot extract descriptors from image %q: %v\n", imageURL2, err))
			return
		}

		names1, names2 := []string{}, []string{}
		for i := range descr1 {
			names1 = append(names1, fmt.Sprintf("Person %d", i))
		}

		for i := range descr2 {
			name := "unknown"
			for j := range descr1 {
				if descr1[j].DistanceTo(descr2[i]) < 0.4 {
					name = fmt.Sprintf("Person %d", j)
				}
			}
			names2 = append(names2, name)
		}

		c.HTML(http.StatusOK, "faces.html", gin.H{
			"imageURL1":     imageURL1,
			"croppedFaces1": allToHTMLBase64(cropped1),
			"descriptors1":  descr1,
			"names1":        names1,
			"imageURL2":     imageURL2,
			"croppedFaces2": allToHTMLBase64(cropped2),
			"descriptors2":  descr2,
			"names2":        names2,
		})
		return
	}
}

func extract(img goimage.Image,
	detector *detection.Detector, landmark *landmarks.Landmark,
	descriptor *descriptors.Descriptor) ([]goimage.Image, []*descriptors.Descriptors, error) {
	allDetections, err := detector.Detect(img)
	if err != nil {
		return nil, nil, errors.Wrap(err, "error detecting faces")
	}

	detections := allDetections.Above(0.5)

	images := []goimage.Image{}
	descrs := []*descriptors.Descriptors{}
	for _, box := range detections.Boxes {
		cropped := goimage.NewRGBA(box)
		draw.Draw(cropped, box, img, box.Min, draw.Src)

		landmarks, err := landmark.Detect(cropped)
		if err != nil {
			return nil, nil, errors.Wrap(err, "error detecting landmarks")
		}

		cropped2 := landmarks.Center(cropped)

		descriptors, err := descriptor.Compute(cropped2)
		if err != nil {
			return nil, nil, errors.Wrap(err, "error computing descriptors")
		}

		images = append(images, cropped2)
		descrs = append(descrs, descriptors)
	}

	return images, descrs, nil
}

func allToHTMLBase64(img []goimage.Image) []template.URL {
	ret := []template.URL{}

	for _, i := range img {
		ret = append(ret, template.URL(toHTMLBase64(i)))
	}

	return ret
}

func toHTMLBase64(img goimage.Image) string {
	var buf bytes.Buffer
	_ = jpeg.Encode(&buf, img, nil)

	b64 := base64.StdEncoding.EncodeToString(buf.Bytes())
	return "data:image/jpeg;base64," + b64
}
