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
	"net/http"
	"sort"
	"strings"

	gildasai "github.com/gildasch/gildas-ai"
	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/gildasch/gildas-ai/sqlite"
	"github.com/gin-gonic/gin"
	"github.com/pkg/errors"
)

func FacesearchHandler(store *sqlite.Store, clusters *FaceClusters) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.HTML(http.StatusOK, "facesearch.html", gin.H{
			"Clusters": clusters.Best(100),
		})
	}
}

func FacesearchDetectionHandler(store *sqlite.Store, clusters *FaceClusters) gin.HandlerFunc {
	return func(c *gin.Context) {
		match := clusters.Find(c.Param("detection"))

		c.HTML(http.StatusOK, "facesearch.html", gin.H{
			"Clusters": []*Matches{match},
			"ShowAll":  true,
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

		match, err := against(store, id1, network1, detectionJSON1, id2, network2, detectionJSON2)
		if err != nil {
			fmt.Println(err)
			c.AbortWithStatus(http.StatusInternalServerError)
			return
		}

		c.HTML(http.StatusOK, "facesearch.html", gin.H{
			"Clusters": []*Matches{match},
			"ShowAll":  true,
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

const (
	threshold = 0.35
)

type FaceClusters struct {
	Clusters map[string]*Matches
}

func (fc *FaceClusters) Best(n int) []*Matches {
	var l []*Matches
	for _, m := range fc.Clusters {
		l = append(l, m)
	}
	sort.Slice(l, func(i, j int) bool { return l[i].Matches > l[j].Matches })

	if len(l) < n {
		return l
	}

	return l[:n]
}

func (fc *FaceClusters) Find(detectionID string) *Matches {
	for _, m := range fc.Clusters {
		if detectionID == m.DetectionID {
			return m
		}

		for _, d := range m.Detections {
			if detectionID == d.DetectionID {
				return m
			}
		}
	}

	return nil
}

type Detection struct {
	DetectionID                string
	ID, Network, DetectionJSON string
	Distance                   float32
	Score                      float32
	Class                      float32
}

type Matches struct {
	Detection
	Matches     int
	AvgDistance float32
	Detections  []Detection
}

func CalculateClusters(store *sqlite.Store) (*FaceClusters, error) {
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

	clusters := &FaceClusters{
		Clusters: map[string]*Matches{},
	}
	whereIs := map[string]string{}
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

		at1, ok1 := whereIs[detectionID1]
		at2, ok2 := whereIs[detectionID2]

		if ok1 && ok2 && at1 == at2 {
			continue
		}

		if distance > threshold {
			if !ok1 {
				clusters.Clusters[detectionID1] = &Matches{
					Detection: Detection{
						DetectionID:   detectionID1,
						ID:            id1,
						Network:       network1,
						DetectionJSON: detectionJSON1,
					},
					Matches:     0,
					AvgDistance: 0,
				}
				whereIs[detectionID1] = detectionID1
			}

			if !ok2 {
				clusters.Clusters[detectionID2] = &Matches{
					Detection: Detection{
						DetectionID:   detectionID2,
						ID:            id2,
						Network:       network2,
						DetectionJSON: detectionJSON2,
					},
					Matches:     0,
					AvgDistance: 0,
				}
				whereIs[detectionID2] = detectionID2
			}

			continue
		}

		if !ok1 && !ok2 {
			clusters.Clusters[detectionID1] = &Matches{
				Detection: Detection{
					DetectionID:   detectionID1,
					ID:            id1,
					Network:       network1,
					DetectionJSON: detectionJSON1,
				},
				Matches:     1,
				AvgDistance: distance,
				Detections: []Detection{
					{
						DetectionID:   detectionID2,
						ID:            id2,
						Network:       network2,
						DetectionJSON: detectionJSON2,
						Distance:      distance,
					},
				},
			}
			whereIs[detectionID1] = detectionID1
			whereIs[detectionID2] = detectionID1

			continue
		}

		if !ok1 {
			clusters.Clusters[at2].Matches++
			clusters.Clusters[at2].AvgDistance += distance
			clusters.Clusters[at2].Detections = append(clusters.Clusters[at2].Detections,
				Detection{
					DetectionID:   detectionID1,
					ID:            id1,
					Network:       network1,
					DetectionJSON: detectionJSON1,
					Distance:      distance,
				})
			whereIs[detectionID1] = at2

			continue
		}

		if !ok2 {
			clusters.Clusters[at1].Matches++
			clusters.Clusters[at1].AvgDistance += distance
			clusters.Clusters[at1].Detections = append(clusters.Clusters[at1].Detections,
				Detection{
					DetectionID:   detectionID2,
					ID:            id2,
					Network:       network2,
					DetectionJSON: detectionJSON2,
					Distance:      distance,
				})
			whereIs[detectionID2] = at1

			continue
		}

		clusters.Clusters[at1].Matches += clusters.Clusters[at2].Matches + 1
		clusters.Clusters[at1].AvgDistance += clusters.Clusters[at2].AvgDistance + distance
		clusters.Clusters[at1].Detections = append(clusters.Clusters[at1].Detections,
			Detection{
				DetectionID:   detectionID2,
				ID:            id2,
				Network:       network2,
				DetectionJSON: detectionJSON2,
				Distance:      distance,
			})
		clusters.Clusters[at1].Detections = append(clusters.Clusters[at1].Detections,
			clusters.Clusters[at2].Detections...)
		whereIs[detectionID2] = at1

		var toUpdate []string
		for id, at := range whereIs {
			if at == at2 {
				whereIs[id] = at1
			}
		}
		for _, id := range toUpdate {
			whereIs[id] = at1
		}

		delete(clusters.Clusters, at2)
	}

	for _, m := range clusters.Clusters {
		m.AvgDistance = m.AvgDistance / float32(m.Matches)
	}

	return clusters, nil
}

func against(store *sqlite.Store, id1, network1, detectionJSON1, id2, network2, detectionJSON2 string) (*Matches, error) {
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

	return &Matches{
		Detection: Detection{
			DetectionID:   makeDetectionID(id1, network1, detectionJSON1),
			ID:            id1,
			Network:       network1,
			DetectionJSON: detectionJSON1,
		},
		Matches:     1,
		AvgDistance: distance,
		Detections: []Detection{
			{
				DetectionID:   makeDetectionID(id2, network2, detectionJSON2),
				ID:            id2,
				Network:       network2,
				DetectionJSON: detectionJSON2,
				Distance:      distance,
			},
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
