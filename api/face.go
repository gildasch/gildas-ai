package api

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/jpeg"
	"net/http"
	"sort"
	"strconv"
	"strings"

	"github.com/gildasch/gildas-ai/faces"
	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/gin-gonic/gin"
	"github.com/nfnt/resize"
	uuid "github.com/satori/go.uuid"
)

func FacesHomeHandler(batches map[string]*faces.Batch) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.HTML(http.StatusOK, "faces.html", gin.H{})
		return
	}
}

func FacesPostBatchHandler(extractor *faces.Extractor, batches map[string]*faces.Batch) gin.HandlerFunc {
	return func(c *gin.Context) {
		fileHeader, err := c.FormFile("image_zip")
		if err != nil {
			c.HTML(http.StatusOK, "faces.html", gin.H{})
			return
		}

		file, err := fileHeader.Open()
		if err != nil {
			c.AbortWithStatusJSON(
				http.StatusBadRequest,
				fmt.Sprintf("cannot open zip file: %v", err))
			return
		}
		defer file.Close()

		images, errs := imageutils.FromZip(file, fileHeader.Size)
		if len(images) == 0 {
			c.AbortWithStatusJSON(
				http.StatusBadRequest,
				fmt.Sprintf("no image found in zip file: %v", errs))
			return
		}

		batch := &faces.Batch{}
		batch = batch.Process(extractor, images)
		id, _ := uuid.NewV4()
		batches[id.String()] = batch

		c.Redirect(http.StatusFound, "/faces/batch/"+id.String())
	}
}

func FacesGetBatchHandler(batches map[string]*faces.Batch) gin.HandlerFunc {
	return func(c *gin.Context) {
		id := c.Param("batchID")

		batch, ok := batches[id]
		if !ok {
			c.AbortWithStatusJSON(
				http.StatusNotFound,
				fmt.Sprintf("batch %q not found", id))
			return
		}

		var matches []match
		cluster := &faceCluster{}

		for i := 0; i < len(batch.Items); i++ {
			cluster.distances = append(cluster.distances, make([]float32, len(batch.Items)))
		}

		for i := 0; i < len(batch.Items); i++ {
			cluster.Images = append(cluster.Images,
				fmt.Sprintf("/faces/batch/%s/cropped/%d.jpg?resize=50", id, i))

			for j := i + 1; j < len(batch.Items); j++ {
				distance, err := batch.Items[i].Descriptors.DistanceTo(batch.Items[j].Descriptors)
				if err != nil {
					fmt.Printf("error calculating distance between %d and %d: %v\n", i, j, err)
					continue
				}

				matches = append(matches, match{
					Name1:    batch.Items[i].Name,
					Name2:    batch.Items[j].Name,
					Cropped1: fmt.Sprintf("/faces/batch/%s/cropped/%d.jpg", id, i),
					Cropped2: fmt.Sprintf("/faces/batch/%s/cropped/%d.jpg", id, j),
					Distance: distance,
				})

				cluster.distances[i][j] = distance
				cluster.distances[j][i] = distance
			}
		}

		sort.Slice(matches, func(i, j int) bool {
			return matches[i].Distance < matches[j].Distance
		})

		var sources []string
		for name := range batch.Sources {
			sources = append(sources, fmt.Sprintf("/faces/batch/%s/sources/%s", id, name))
		}

		cluster.Points = project2D(cluster)

		c.HTML(http.StatusOK, "faces.html", gin.H{
			"sources": sources,
			"matches": matches,
			"cluster": cluster,
		})
		return
	}
}

type match struct {
	Name1, Name2       string
	Cropped1, Cropped2 string
	Distance           float32
}

type faceCluster struct {
	Images    []string
	distances [][]float32
	Points    []point
}

func (f *faceCluster) Len() int {
	return len(f.Images)
}

func (f *faceCluster) Distance(i, j int) float32 {
	return 1000 * f.distances[i][j]
}

func toHTMLBase64(img image.Image) string {
	var buf bytes.Buffer
	_ = jpeg.Encode(&buf, img, nil)

	b64 := base64.StdEncoding.EncodeToString(buf.Bytes())
	return "data:image/jpeg;base64," + b64
}

func FaceSourceHandler(batches map[string]*faces.Batch) gin.HandlerFunc {
	return func(c *gin.Context) {
		batchID := c.Param("batchID")
		name := c.Param("name")

		source := batches[batchID].Sources[name]

		var jpegBytes bytes.Buffer
		err := jpeg.Encode(&jpegBytes, source, nil)
		if err != nil {
			c.AbortWithStatusJSON(
				http.StatusInternalServerError,
				fmt.Sprintf("error writing out image: %v", err))
			return
		}

		c.Data(http.StatusOK, "image/jpeg", jpegBytes.Bytes())
	}
}

func FaceCroppedHandler(batches map[string]*faces.Batch) gin.HandlerFunc {
	return func(c *gin.Context) {
		batchID := c.Param("batchID")
		name := c.Param("name")

		var i int
		var err error
		if i, err = strconv.Atoi(strings.TrimSuffix(name, ".jpg")); err != nil || i >= len(batches[batchID].Items) {
			c.AbortWithStatusJSON(
				http.StatusBadRequest,
				fmt.Sprintf("error parsing image name %q: %v", name, err))
			return
		}

		cropped := batches[batchID].Items[i].Cropped

		if maxStr := c.Query("resize"); maxStr != "" {
			if max, err := strconv.Atoi(maxStr); err == nil {
				cropped = resize.Resize(uint(max), uint(max), cropped, resize.NearestNeighbor)
			}
		}

		var jpegBytes bytes.Buffer
		err = jpeg.Encode(&jpegBytes, cropped, nil)
		if err != nil {
			c.AbortWithStatusJSON(
				http.StatusInternalServerError,
				fmt.Sprintf("error writing out image: %v", err))
			return
		}

		c.Data(http.StatusOK, "image/jpeg", jpegBytes.Bytes())
	}
}
