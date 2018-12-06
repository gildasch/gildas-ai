package main

import (
	"bufio"
	"fmt"
	"image"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/gildasch/gildas-ai/tensor"
)

const threshold = 0.1

func usage() {
	fmt.Printf("%s [model-root-folder] [image-folder]\n", os.Args[0])
}

type Classifier interface {
	Inception(img image.Image) (*tensor.Predictions, error)
}

func main() {
	if len(os.Args) < 3 {
		usage()
		return
	}

	modelRootFolder := os.Args[1]
	imageFolder := os.Args[2]

	resnet := &tensor.Model{
		ModelName:   modelRootFolder + "/resnet",
		TagName:     "myTag",
		InputLayer:  "input_1",
		OutputLayer: "fc1000/Softmax",
		ImageMode:   tensor.ImageModeCaffe,
		Labels:      "imagenet_class_index.json",
		ImageHeight: 224,
		ImageWidth:  224,
	}
	close, err := resnet.Load()
	if err != nil {
		log.Fatal("could not load classifier", err)
	}
	defer func() {
		if err := close(); err != nil {
			fmt.Println("error closing classifier:", err)
		}
	}()

	objects, err := inspectFolder(resnet, imageFolder)
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

func inspectFolder(classifier Classifier, folder string) (map[string][]string, error) {
	files, err := filepath.Glob(strings.TrimSuffix(folder, "/") + "/*")
	if err != nil {
		return nil, err
	}

	objects := map[string][]string{}
	for i, file := range files {
		fmt.Printf("\r(%d/%d) processing %s", i, len(files), file)

		img, err := imageutils.FromFile(file)
		if err != nil {
			fmt.Printf("\nerror processing file %s: %v\n", file, err)
			continue
		}

		preds, err := classifier.Inception(img)
		if err != nil {
			fmt.Printf("\nerror executing inception on %s: %v\n", file, err)
			continue
		}

		for _, p := range preds.Above(threshold) {
			label := strings.ToLower(p.Label)
			objects[label] = append(objects[label], file)
		}
	}
	fmt.Println()

	return objects, nil
}

func find(objects map[string][]string, query string) []string {
	query = strings.ToLower(strings.TrimSuffix(query, "\n"))
	return objects[query]
}
