package main

import (
	"encoding/json"
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
	fmt.Printf("%s [model-root-folder] [faces-folder] [face-to-recognive]\n", os.Args[0])
}

func main() {
	if len(os.Args) < 4 {
		usage()
		return
	}

	modelRootFolder := os.Args[1]
	facesFolder := os.Args[2]
	faceToRecognize := os.Args[3]

	noCalculation := false
	if len(os.Args) >= 5 && os.Args[4] == "-no-calculation" {
		noCalculation = true
	}

	extractor, err := faceapi.NewDefaultExtractor(modelRootFolder)
	if err != nil {
		log.Fatal(err)
	}

	targetImg, err := imageutils.FromFile(faceToRecognize)
	if err != nil {
		fmt.Println(err)
	}

	_, targetDescr, err := extractor.Extract(targetImg)
	if err != nil {
		log.Fatal(err)
	}

	if len(targetDescr) < 1 {
		fmt.Printf("no face found in %s", os.Args[1])
		return
	}

	fmt.Printf("%d face(s) found in %s\n", len(targetDescr), os.Args[1])

	descrs, err := calculateDescriptors(extractor, facesFolder, noCalculation)
	if err != nil {
		log.Fatal(err)
	}

	for name, descr := range descrs {
		score, err := targetDescr[0].DistanceTo(*descr)
		if err != nil {
			fmt.Println("error calculating distance", err)
			continue
		}
		if score < 0.4 {
			fmt.Println(score, name)
		}
	}
}

func calculateDescriptors(extractor *gildasai.Extractor,
	facesFolder string, noCalculation bool) (map[string]*gildasai.Descriptors, error) {
	faceFiles, err := filepath.Glob(strings.TrimSuffix(facesFolder, "/") + "/*")
	if err != nil {
		return nil, err
	}

	descrs := preCalculatedOrNew(facesFolder)

	if noCalculation {
		return descrs, nil
	}

	for _, faceFile := range faceFiles {
		if strings.Contains(faceFile, ".cropped.jpg") {
			continue
		}

		if _, ok := descrs[fmt.Sprintf("%s/%d", faceFile, 0)]; ok {
			fmt.Printf("skipping already-processed %s\n", faceFile)
			continue
		}

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
			descrs[fmt.Sprintf("%s/%d", faceFile, i)] = &d
			saveImage(fmt.Sprintf("%s.%d", faceFile, i), ii[i])
		}
	}

	savePreCalculated(facesFolder, descrs)

	return descrs, nil
}

func preCalculatedOrNew(facesFolder string) map[string]*gildasai.Descriptors {
	ds := preCalculated(facesFolder)
	if ds != nil {
		return ds
	}
	return map[string]*gildasai.Descriptors{}
}

func preCalculated(facesFolder string) map[string]*gildasai.Descriptors {
	f, err := os.Open(strings.TrimSuffix(facesFolder, "/") + "/precalculated.json")
	if err != nil {
		return nil
	}

	var descrs map[string]*gildasai.Descriptors
	err = json.NewDecoder(f).Decode(&descrs)
	if err != nil {
		return nil
	}

	return descrs
}

func savePreCalculated(facesFolder string, descrs map[string]*gildasai.Descriptors) {
	f, err := os.Create(strings.TrimSuffix(facesFolder, "/") + "/precalculated.json")
	if err != nil {
		return
	}

	err = json.NewEncoder(f).Encode(descrs)
	if err != nil {
		return
	}

	return
}

func saveImage(filename string, img image.Image) {
	f, err := os.Create(filename + ".cropped.jpg")
	if err != nil {
		fmt.Printf("error saving %q: %v", filename, err)
		return
	}
	jpeg.Encode(f, img, nil)
}
