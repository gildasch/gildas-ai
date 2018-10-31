package main

import (
	"fmt"
	goimage "image"
	"os"
	"strings"

	"github.com/gildasch/gildas-ai/image"
	"github.com/gildasch/gildas-ai/tensor"
)

func usage() {
	fmt.Printf("Usage: %s [xception|resnet] path/to/image.jpg\n", os.Args[0])
}

func main() {
	if len(os.Args) < 3 {
		usage()
		return
	}
	modelName := os.Args[1]
	imageName := os.Args[2]

	var model *tensor.Model
	var err error
	switch modelName {
	case "xception":
		var close func() error
		model, close, err = tensor.NewModel("myModel", "myTag", "input_1", "predictions/Softmax", tensor.ImageModeTensorflow, "imagenet_class_index.json")
		if err != nil {
			fmt.Printf("Error loading saved model: %s\n", err.Error())
			return
		}
		defer close()
	case "resnet":
		var close func() error
		model, close, err = tensor.NewModel("resnet", "myTag", "input_1", "fc1000/Softmax", tensor.ImageModeCaffe, "imagenet_class_index.json")
		if err != nil {
			fmt.Printf("Error loading saved model: %s\n", err.Error())
			return
		}
		defer close()
	default:
		usage()
		return
	}

	var img goimage.Image
	if strings.HasPrefix(imageName, "https://") || strings.HasPrefix(imageName, "http://") {
		img, err = image.FromURL(imageName)
		if err != nil {
			fmt.Printf("cannot read remote image %q: %v\n", imageName, err)
			return
		}
	} else {
		img, err = image.FromFile(imageName)
		if err != nil {
			fmt.Printf("cannot read local image %q: %v\n", imageName, err)
			return
		}
	}

	preds, err := model.Inception(img)
	if err != nil {
		fmt.Printf("there was an error while running the inception: %v\n", err)
		return
	}

	bests := preds.Best(10)
	fmt.Println("Results:")
	for _, b := range bests {
		fmt.Printf("%v (%f)\n", b.Label, b.Score)
	}
}
