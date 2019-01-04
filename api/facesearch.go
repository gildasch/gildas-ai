package api

import (
	"bytes"
	"fmt"
	"image"
	"image/draw"
	"image/jpeg"
	"log"
	"net/http"
	"strconv"

	gildasai "github.com/gildasch/gildas-ai"
	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/gin-gonic/gin"
)

const (
	detectionThreshold = 0.98
	threshold          = 0.37
)

var matches [10]faceMatchings

func FacesearchHandler(store gildasai.FaceStore) gin.HandlerFunc {
	faceItems, err := store.GetAllFaces()
	if err != nil {
		log.Fatal("could not get all faces from store: ", err)
	}

	for counter, fi := range faceItems {
		if fi.Detection.Score < detectionThreshold {
			continue
		}

		var currentMatches []int
		var distances []float32
		var scores []float32
		var ids []string
		for i, f := range faceItems {
			if f.Detection.Score < detectionThreshold {
				continue
			}

			dist, err := fi.Descriptors.DistanceTo(f.Descriptors)
			if err != nil {
				continue
			}
			if dist == 0 {
				continue
			}
			if dist > threshold {
				continue
			}
			currentMatches = append(currentMatches, i)
			distances = append(distances, dist)
			scores = append(scores, f.Detection.Score)
			ids = append(ids, f.Identifier)
		}

		if isKnown(fi.Identifier) {
			continue
		}

		lessMatches := findLessMatches(len(currentMatches))
		if lessMatches == -1 {
			continue
		}

		fmt.Printf("found a matches that is lower that %d: %d at %d\n",
			len(currentMatches), matches[lessMatches].Matches, lessMatches)

		if len(distances) >= 51 {
			distances = distances[:51]
			scores = scores[:51]
		}

		matches[lessMatches] = faceMatchings{
			ID:        fi.Identifier,
			Score:     fi.Detection.Score,
			Matches:   len(currentMatches),
			Distances: distances,
			Scores:    scores,
			IDs:       ids,
		}

		matches[lessMatches].Cropped = crop(fi.Identifier, fi.Detection.Box)

		for mi, m := range currentMatches {
			fmt.Printf("\rcalculating crops of %d (%d/%d)", counter, mi, len(currentMatches))
			matches[lessMatches].CroppedMatches = append(matches[lessMatches].CroppedMatches,
				crop(faceItems[m].Identifier, faceItems[m].Detection.Box))
			if mi > 50 {
				break
			}
		}
		fmt.Println()
	}

	return func(c *gin.Context) {
		c.HTML(http.StatusOK, "facesearch.html", gin.H{
			"Matches": matches,
		})
	}
}

func FacesearchImageHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		face, _ := strconv.Atoi(c.Param("face"))
		c.Data(200, "image/jpeg", matches[face].Cropped)
	}
}

func FacesearchImageMatchHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		face, _ := strconv.Atoi(c.Param("face"))
		match, _ := strconv.Atoi(c.Param("match"))
		c.Data(200, "image/jpeg", matches[face].CroppedMatches[match])
	}
}

type faceMatchings struct {
	ID      string
	Score   float32
	Cropped []byte

	Matches        int
	Distances      []float32
	Scores         []float32
	IDs            []string
	CroppedMatches [][]byte
}

func crop(filename string, box image.Rectangle) []byte {
	img, err := imageutils.FromFile(filename)
	if err != nil {
		return []byte{}
	}

	out := image.NewRGBA(box)
	draw.Draw(out, out.Bounds(), img, box.Min, draw.Src)

	var b bytes.Buffer
	err = jpeg.Encode(&b, out, nil)
	if err != nil {
		return []byte{}
	}

	return b.Bytes()
}

func findLessMatches(n int) int {
	min, minIndex := matches[0].Matches, 0

	for i, m := range matches {
		if m.Matches < min {
			min = m.Matches
			minIndex = i
		}
	}

	if min < n {
		return minIndex
	}
	return -1
}

func isKnown(id string) bool {
	for _, m := range matches {
		for _, known := range m.IDs {
			if known == id {
				return true
			}
		}
	}
	return false
}
