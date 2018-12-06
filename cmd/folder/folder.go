package main

import (
	"bufio"
	"fmt"
	"image"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/gildasch/gildas-ai/cmd/folder/cache/localcache"
	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/gildasch/gildas-ai/tensor"
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
	Inception(img image.Image) (*tensor.Predictions, error)
}

type Cache interface {
	Inception(file string, inception func() ([]string, error)) ([]string, error)
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
		pnasnet := &tensor.Model{
			ModelName:       modelRootFolder + "/pnasnet",
			TagName:         "myTag",
			InputLayer:      "module/hub_input/images",
			OutputLayer:     "module/final_layer/predictions",
			ImageMode:       tensor.ImageModeTensorflowPositive,
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
		localCache := &localcache.LocalCache{
			CacheFile: imageFolder + "/.inception.json",
		}
		cache = localCache
	}

	objects, err := inspectFolder(cache, classifier, imageFolder)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(objects)

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

		var inception func() ([]string, error)
		if classifier != nil {
			inception = func() ([]string, error) {
				predictions, err := classifier.Inception(img)
				if err != nil {
					return nil, errors.Wrapf(err, "error executing inception on %s", file)
				}

				preds := []string{}
				for _, p := range predictions.Above(threshold) {
					preds = append(preds, strings.ToLower(p.Label))
				}

				return preds, nil
			}
		} else {
			inception = func() ([]string, error) {
				return nil, errors.New("no classifier given")
			}
		}

		var preds []string
		if cache != nil {
			preds, err = cache.Inception(file, inception)
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
			objects[p] = append(objects[p], file)
		}
	}
	fmt.Println()

	return objects, nil
}

func find(objects map[string][]string, query string) []string {
	query = strings.ToLower(strings.TrimSuffix(query, "\n"))
	return objects[query]
}
