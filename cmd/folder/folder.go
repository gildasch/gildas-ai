package main

import (
	"bufio"
	"fmt"
	"image"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/gildasch/gildas-ai/cmd/folder/cache/sqlite"
	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/gildasch/gildas-ai/objects/classifiers"
	"github.com/pkg/errors"
)

const (
	threshold     = 0.1
	noCache       = false
	onlyFromCache = false
)

func usage() {
	fmt.Printf("%s [model-root-folder] [image-folder]\n", os.Args[0])
}

type Classifier interface {
	Inception(img image.Image) (*classifiers.Predictions, error)
}

type Cache interface {
	Inception(file, network string, inception func() ([]classifiers.Prediction, error)) ([]classifiers.Prediction, error)
}

func main() {
	if len(os.Args) < 3 {
		usage()
		return
	}

	modelRootFolder := strings.TrimSuffix(os.Args[1], "/")
	imageFolder := strings.TrimSuffix(os.Args[2], "/")

	var classifier Classifier
	if !onlyFromCache {
		pnasnet := &classifiers.Model{
			ModelName:       modelRootFolder + "/pnasnet",
			TagName:         "myTag",
			InputLayer:      "module/hub_input/images",
			OutputLayer:     "module/final_layer/predictions",
			ImageMode:       classifiers.ImageModeTensorflowPositive,
			Labels:          "imagenet_class_index.json",
			ImageHeight:     331,
			ImageWidth:      331,
			IndexCorrection: -1,
		}
		close, err := pnasnet.Load()
		if err != nil {
			log.Fatal("could not load classifier", err)
		}
		defer func() {
			if err := close(); err != nil {
				fmt.Println("error closing classifier:", err)
			}
		}()
		classifier = pnasnet
	}

	var cache Cache
	if !noCache {
		sqliteCache, err := sqlite.NewCache(imageFolder + "/.inception.sqlite")
		if err != nil {
			log.Fatal(err)
		}

		cache = sqliteCache
	}

	objects, err := inspectFolder(cache, classifier, imageFolder)
	if err != nil {
		log.Fatal(err)
	}

	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Printf("search: ")
		query, _ := reader.ReadString('\n')

		fmt.Println()
		for _, result := range find(objects, query) {
			fmt.Println(result)
		}
	}
}

func inspectFolder(cache Cache, classifier Classifier, folder string) (map[string][]string, error) {
	if cache == nil && classifier == nil {
		return nil, errors.New("cannot inspect without cache or classifier")
	}

	files, err := filepath.Glob(folder + "/*")
	if err != nil {
		return nil, err
	}

	objects := map[string][]string{}
	for i, file := range files {
		fmt.Printf("(%d/%d) processing %s\n", i+1, len(files), file)

		img, err := imageutils.FromFile(file)
		if err != nil {
			fmt.Printf("error processing file %s: %v\n", file, err)
			continue
		}

		var inception func() ([]classifiers.Prediction, error)
		if classifier != nil {
			inception = func() ([]classifiers.Prediction, error) {
				predictions, err := classifier.Inception(img)
				if err != nil {
					return nil, errors.Wrapf(err, "error executing inception on %s", file)
				}

				return predictions.Best(10), nil
			}
		} else {
			inception = func() ([]classifiers.Prediction, error) {
				return nil, errors.New("no classifier given")
			}
		}

		var preds []classifiers.Prediction
		if cache != nil {
			preds, err = cache.Inception(file, "pnasnet", inception)
			if err != nil {
				fmt.Printf("%v\n", err)
				continue
			}
		} else {
			preds, err = inception()
			if err != nil {
				fmt.Printf("error executing inception on %s: %v\n", file, err)
				continue
			}
		}

		for _, p := range preds {
			objects[p.Label] = append(objects[p.Label], file)
		}
	}
	fmt.Println()

	return objects, nil
}

func find(objects map[string][]string, query string) []string {
	query = strings.ToLower(strings.TrimSuffix(query, "\n"))
	return objects[query]
}
