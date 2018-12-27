package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"log"
	"os"
	"path/filepath"
	"strings"

	gildasai "github.com/gildasch/gildas-ai"
	"github.com/gildasch/gildas-ai/faceapi"
	"github.com/gildasch/gildas-ai/imageutils"
)

func usage() {
	fmt.Printf("%s [model-root-folder] [faces-folder]\n", os.Args[0])
}

func main() {
	if len(os.Args) < 3 {
		usage()
		return
	}

	modelRootFolder := os.Args[1]
	facesFolder := os.Args[2]

	extractor, err := faceapi.NewDefaultExtractor(modelRootFolder)
	if err != nil {
		log.Fatal(err)
	}

	descrs, err := calculateDescriptors(extractor, facesFolder)
	if err != nil {
		log.Fatal(err)
	}

	clusters := calculateClusters(descrs, 0.25)

	fmt.Println(clusters)
}

func calculateDescriptors(extractor *gildasai.Extractor,
	facesFolder string) (map[string]*gildasai.Descriptors, error) {
	faceFiles, err := filepath.Glob(strings.TrimSuffix(facesFolder, "/") + "/*")
	if err != nil {
		return nil, err
	}

	descrs := map[string]*gildasai.Descriptors{}
	for _, faceFile := range faceFiles {
		fmt.Printf("processing %s\n", faceFile)

		img, err := imageutils.FromFile(faceFile)
		if err != nil {
			fmt.Printf("error processing file %s: %v\n", faceFile, err)
			continue
		}

		ii, dd, err := extractor.Extract(img)
		if err != nil {
			fmt.Printf("error extracting from %s: %v\n", faceFile, err)
			continue
		}

		if len(dd) < 1 {
			fmt.Printf("no face found in %s: %v\n", faceFile, err)
			continue
		}

		for i, d := range dd {
			descrs[fmt.Sprintf("%s.%d", faceFile, i)] = &d
			saveImage(fmt.Sprintf("%s.%d", faceFile, i), ii[i])
		}
	}

	return descrs, nil
}

func saveImage(filename string, img image.Image) {
	f, err := os.Create(filename + ".cropped.jpg")
	if err != nil {
		fmt.Printf("error saving %q: %v", filename, err)
		return
	}
	jpeg.Encode(f, img, nil)
}

func calculateClusters(descrs map[string]*gildasai.Descriptors, threshold float32) [][]string {
	clusters := [][]string{}
descrsloop:
	for name, descr := range descrs {
		for i, cc := range clusters {
			for _, c := range cc {
				dist, err := descr.DistanceTo(*descrs[c])
				if err != nil {
					fmt.Println("error calculating distance:", err)
					continue
				}
				if dist < threshold {
					clusters[i] = append(clusters[i], name)
					continue descrsloop
				}
			}
		}
		clusters = append(clusters, []string{name})
	}

	return clusters
}
