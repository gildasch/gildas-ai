package api

import (
	"bytes"
	"database/sql"
	"encoding/base32"
	"encoding/json"
	"fmt"
	"image"
	"image/draw"
	"image/jpeg"
	"log"
	"net/http"
	"strings"

	gildasai "github.com/gildasch/gildas-ai"
	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/gildasch/gildas-ai/sqlite"
	"github.com/gin-gonic/gin"
	"github.com/pkg/errors"
)

const (
	threshold = 0.37
)

func FacesearchHandler(store *sqlite.Store) gin.HandlerFunc {
	detections, err := findBestMatches(store)
	if err != nil {
		log.Fatal("could not get best matches from store: ", err)
	}

	return func(c *gin.Context) {
		c.HTML(http.StatusOK, "facesearch.html", gin.H{
			"Detections": detections,
		})
	}
}

func FacesearchDetectionImageHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		id, _, detectionJSON, err := readDetectionID(c.Param("detection"))
		if err != nil {
			c.AbortWithStatus(http.StatusBadRequest)
			return
		}

		var detection gildasai.Detection
		err = json.Unmarshal([]byte(detectionJSON), &detection)
		if err != nil {
			c.AbortWithStatus(http.StatusBadRequest)
			return
		}

		c.Data(200, "image/jpeg", crop(id, detection.Box))
	}
}

func FacesearchLandmarkImageHandler(store *sqlite.Store) gin.HandlerFunc {
	return func(c *gin.Context) {
		id, network, detectionJSON, err := readDetectionID(c.Param("detection"))
		if err != nil {
			c.AbortWithStatus(http.StatusBadRequest)
			return
		}

		var landmarksJSON string
		err = store.QueryRow(`
select landmarks from faces
where id = $1 and network = $2 and detection = $3
`, id, network, detectionJSON).Scan(&landmarksJSON)
		if err == sql.ErrNoRows {
			c.AbortWithStatus(http.StatusNotFound)
			return
		}
		if err != nil {
			fmt.Println(err)
			c.AbortWithStatus(http.StatusBadRequest)
			return
		}

		var landmarks gildasai.Landmarks
		err = json.Unmarshal([]byte(landmarksJSON), &landmarks)
		if err != nil {
			c.AbortWithStatus(http.StatusInternalServerError)
			return
		}

		c.Data(200, "image/jpeg", landmarksImage(landmarks))
	}
}

type detection struct {
	DetectionID string
	Matches     int
	AvgDistance float32
}

func findBestMatches(store *sqlite.Store) ([]detection, error) {
	rows, err := store.Query(`
select id, network, detection, count(*) as matches, avg(distance) as avg_distance
from faces
  join face_distances on (
    (id = id1 or id = id2)
    and (network = network1 or network = network2)
    and (detection = detection1 or detection = detection2))
where distance != 0
  and distance < $1
group by id, network, detection
order by matches desc, avg_distance
limit 100
`, threshold)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var detections []detection
	for rows.Next() {
		var id, network, detectionJSON string
		var matches int
		var avgDistance float32
		err := rows.Scan(&id, &network, &detectionJSON, &matches, &avgDistance)
		if err != nil {
			return nil, err
		}

		detections = append(detections, detection{
			DetectionID: makeDetectionID(id, network, detectionJSON),
			Matches:     matches,
			AvgDistance: avgDistance,
		})
	}

	return detections, nil
}

func makeDetectionID(id, network, detectionJSON string) string {
	return base32.StdEncoding.EncodeToString([]byte(id + "|" + network + "|" + detectionJSON))
}

func readDetectionID(detectionID string) (id, network, detectionJSON string, err error) {
	data, err := base32.StdEncoding.DecodeString(detectionID)
	if err != nil {
		return "", "", "", err
	}

	splitted := strings.Split(string(data), "|")
	if len(splitted) != 3 {
		return "", "", "", errors.Errorf("expected 3 parts in detectionID, found %d", len(splitted))
	}

	return splitted[0], splitted[1], splitted[2], nil
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

func landmarksImage(landmarks gildasai.Landmarks) []byte {
	out := landmarks.DrawOnImage(image.NewRGBA(image.Rect(0, 0, 200, 200)))

	var b bytes.Buffer
	err := jpeg.Encode(&b, out, nil)
	if err != nil {
		return []byte{}
	}

	return b.Bytes()
}
