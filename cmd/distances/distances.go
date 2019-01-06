package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"

	gildasai "github.com/gildasch/gildas-ai"
	"github.com/gildasch/gildas-ai/sqlite"
)

const (
	maxDistance = 0.6
)

func usage() {
	fmt.Printf("%s [sqlite-db-file]\n", os.Args[0])
}

func main() {
	if len(os.Args) < 2 {
		usage()
		return
	}

	sqliteFile := os.Args[1]

	store, err := sqlite.NewStore(sqliteFile)
	if err != nil {
		log.Fatal("could not create store with file "+sqliteFile+": ", err)
	}
	defer store.Close()

	faceItems, err := store.GetAllFaces()
	if err != nil {
		log.Fatal("could not get all faces from store: ", err)
	}

	total := len(faceItems)
	for i1, fi1 := range faceItems {
		if len(fi1.Descriptors) == 0 || fi1.Detection.Score < 0.9 || fi1.Landmarks.Confidence() < 0.5 {
			continue
		}
		for i2, fi2 := range faceItems[i1+1:] {
			if len(fi2.Descriptors) == 0 || fi2.Detection.Score < 0.9 || fi2.Landmarks.Confidence() < 0.5 {
				continue
			}
			fmt.Printf("\rprogress: %d/%d (distance with: %d/%d)", i1, total, i2, total)

			fi1, fi2 := sort(fi1, fi2)

			_, ok, err := store.GetFaceDistance(fi1, fi2)
			if err != nil {
				log.Fatal("could not get distance from store: ", err)
			}
			if ok {
				continue
			}

			dist, err := fi1.Descriptors.DistanceTo(fi2.Descriptors)
			if err != nil {
				log.Fatal("could not calculate distance: ", err)
			}

			if dist > maxDistance {
				continue
			}

			err = store.StoreFaceDistance(fi1, fi2, dist)
			if err != nil {
				log.Fatal("could not store distance to store: ", err)
			}
		}
	}
}

func sort(fi1, fi2 *gildasai.FaceItem) (s1, s2 *gildasai.FaceItem) {
	detection1, err := json.Marshal(fi1.Detection)
	if err != nil {
		return fi1, fi2
	}
	detection2, err := json.Marshal(fi2.Detection)
	if err != nil {
		return fi1, fi2
	}

	str1 := fi1.Identifier + fi1.Network + string(detection1)
	str2 := fi2.Identifier + fi2.Network + string(detection2)

	if str1 <= str2 {
		return fi1, fi2
	}

	return fi2, fi1
}
