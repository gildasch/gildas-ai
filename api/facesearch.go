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
	"sort"
	"strings"

	gildasai "github.com/gildasch/gildas-ai"
	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/gildasch/gildas-ai/sqlite"
	"github.com/gin-gonic/gin"
	"github.com/pkg/errors"
)

const (
	threshold = 0.40
)

func FacesearchHandler(store *sqlite.Store) gin.HandlerFunc {
	detections, err := findBestMatches(store)
	if err != nil {
		log.Fatal("could not get best matches from store: ", err)
	}

	return func(c *gin.Context) {
		if len(detections) > 100 {
			detections = detections[:100]
		}
		c.HTML(http.StatusOK, "facesearch.html", gin.H{
			"Detections": detections,
		})
	}
}

func FacesearchDetectionHandler(store *sqlite.Store) gin.HandlerFunc {
	return func(c *gin.Context) {
		id, network, detectionJSON, err := readDetectionID(c.Param("detection"))
		if err != nil {
			c.AbortWithStatus(http.StatusBadRequest)
			return
		}

		detections, err := findMatches(store, id, network, detectionJSON)
		if err != nil {
			fmt.Println(err)
			c.AbortWithStatus(http.StatusInternalServerError)
			return
		}

		c.HTML(http.StatusOK, "facesearch.html", gin.H{
			"Detections": detections,
		})
	}
}

func FacesearchAgainstHandler(store *sqlite.Store) gin.HandlerFunc {
	return func(c *gin.Context) {
		id1, network1, detectionJSON1, err := readDetectionID(c.Param("detection"))
		if err != nil {
			c.AbortWithStatus(http.StatusBadRequest)
			return
		}

		id2, network2, detectionJSON2, err := readDetectionID(c.Param("detection2"))
		if err != nil {
			c.AbortWithStatus(http.StatusBadRequest)
			return
		}

		detections, err := against(store, id1, network1, detectionJSON1, id2, network2, detectionJSON2)
		if err != nil {
			fmt.Println(err)
			c.AbortWithStatus(http.StatusInternalServerError)
			return
		}

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
	DetectionID                string
	ID, Network, DetectionJSON string
	Score                      float32
	Class                      float32
	Matches                    int
	AvgDistance                float32
	Distance                   float32
}

type detectionsCluster struct {
	Items   []*detection
	whereIs map[string]int
}

func findBestMatches(store *sqlite.Store) ([]detection, error) {
	rows, err := store.Query(`
select id1, network1, detection1, id2, network2, detection2, distance
from face_distances
where distance != 0
order by distance
`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var cluster detectionsCluster
	cluster.whereIs = map[string]int{}
	for rows.Next() {
		var id1, network1, detectionJSON1 string
		var id2, network2, detectionJSON2 string
		var distance float32
		err := rows.Scan(
			&id1, &network1, &detectionJSON1,
			&id2, &network2, &detectionJSON2,
			&distance)
		if err != nil {
			return nil, err
		}

		detectionID1 := makeDetectionID(id1, network1, detectionJSON1)
		detectionID2 := makeDetectionID(id2, network2, detectionJSON2)

		at1, ok1 := cluster.whereIs[detectionID1]
		at2, ok2 := cluster.whereIs[detectionID2]

		if ok1 && ok2 && at1 == at2 {
			continue
		}

		if distance > threshold {
			if !ok1 {
				cluster.Items = append(cluster.Items, &detection{
					DetectionID:   detectionID1,
					ID:            id1,
					Network:       network1,
					DetectionJSON: detectionJSON1,
					Matches:       0,
					AvgDistance:   0,
				})
				cluster.whereIs[detectionID1] = len(cluster.Items) - 1
			}

			if !ok2 {
				cluster.Items = append(cluster.Items, &detection{
					DetectionID:   detectionID2,
					ID:            id2,
					Network:       network2,
					DetectionJSON: detectionJSON2,
					Matches:       0,
					AvgDistance:   0,
				})
				cluster.whereIs[detectionID2] = len(cluster.Items) - 1
			}

			continue
		}

		if !ok1 && !ok2 {
			cluster.Items = append(cluster.Items, &detection{
				DetectionID:   detectionID1,
				ID:            id1,
				Network:       network1,
				DetectionJSON: detectionJSON1,
				Matches:       1,
				AvgDistance:   distance,
			})
			cluster.whereIs[detectionID1] = len(cluster.Items) - 1
			cluster.whereIs[detectionID2] = len(cluster.Items) - 1

			continue
		}

		if !ok1 {
			cluster.Items[at2].Matches++
			cluster.Items[at2].AvgDistance += distance
			cluster.whereIs[detectionID1] = at2

			continue
		}

		if !ok2 {
			cluster.Items[at1].Matches++
			cluster.Items[at1].AvgDistance += distance
			cluster.whereIs[detectionID2] = at1

			continue
		}

		cluster.Items[at1].Matches += cluster.Items[at2].Matches + 1
		cluster.Items[at1].AvgDistance += cluster.Items[at2].AvgDistance + distance

		var toUpdate []string
		for id, at := range cluster.whereIs {
			if at == at2 {
				toUpdate = append(toUpdate, id)
			}
		}
		for _, id := range toUpdate {
			cluster.whereIs[id] = at1
		}

		cluster.Items[at2] = nil
	}

	fmt.Println(len(cluster.Items))
	fmt.Println(cluster.Items[:10])

	var detections []detection
	for _, d := range cluster.Items {
		if d == nil {
			continue
		}

		if d.Matches > 0 {
			d.AvgDistance = d.AvgDistance / float32(d.Matches)
		}

		detections = append(detections, *d)
	}

	fmt.Println(len(detections))

	sort.Slice(detections, func(i, j int) bool { return detections[i].Matches > detections[j].Matches })

	return detections, nil
}

func findMatches(store *sqlite.Store, id, network, detectionJSON string) ([]detection, error) {
	rows, err := store.Query(`
select id, network, detection, 0
from faces
where id = $1 and network = $2 and detection = $3
union all
select id, network, detection, avg(distance) as avg_distance
from faces
  join face_distances on (
    (id = id1 and network = network1 and detection = detection1)
    or (id = id2 and network = network2 and detection = detection2))
where (id1 = $1 or id2 = $1)
  and (network1 = $2 or network2 = $2)
  and (detection1 = $3 or detection2 = $3)
  and distance < $4
group by id, network, detection
order by avg_distance
`, id, network, detectionJSON, threshold)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var detections []detection
	for rows.Next() {
		var id, network, detectionJSON string
		var distance float32
		err := rows.Scan(&id, &network, &detectionJSON, &distance)
		if err != nil {
			return nil, err
		}

		var d gildasai.Detection
		err = json.Unmarshal([]byte(detectionJSON), &d)
		if err != nil {
			return nil, err
		}

		detections = append(detections, detection{
			DetectionID: makeDetectionID(id, network, detectionJSON),
			ID:          id,
			Score:       d.Score,
			Class:       d.Class,
			Distance:    distance,
		})
	}

	return detections, nil
}

func against(store *sqlite.Store, id1, network1, detectionJSON1, id2, network2, detectionJSON2 string) ([]detection, error) {
	var descrsJSON1 string
	err := store.QueryRow(`
select descriptors
from faces
where id = $1 and network = $2 and detection = $3
`, id1, network1, detectionJSON1).Scan(&descrsJSON1)
	if err != nil {
		return nil, err
	}
	var descrs1 gildasai.Descriptors
	err = json.Unmarshal([]byte(descrsJSON1), &descrs1)
	if err != nil {
		return nil, err
	}

	var descrsJSON2 string
	err = store.QueryRow(`
select descriptors
from faces
where id = $1 and network = $2 and detection = $3
`, id2, network2, detectionJSON2).Scan(&descrsJSON2)
	if err != nil {
		return nil, err
	}
	var descrs2 gildasai.Descriptors
	err = json.Unmarshal([]byte(descrsJSON2), &descrs2)
	if err != nil {
		return nil, err
	}

	distance, err := descrs1.DistanceTo(descrs2)
	if err != nil {
		return nil, err
	}

	return []detection{
		{
			DetectionID: makeDetectionID(id1, network1, detectionJSON1),
			ID:          id1,
			Distance:    distance,
		},
		{
			DetectionID: makeDetectionID(id2, network2, detectionJSON2),
			ID:          id2,
			Distance:    distance,
		},
	}, nil
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
