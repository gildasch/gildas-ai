package main

import (
	"fmt"
	goimage "image"
	"os"
	"strings"

	"github.com/gildasch/gildas-ai/image"
	"github.com/gildasch/gildas-ai/tensor"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Printf("Usage: %s path/to/image.jpg\n", os.Args[0])
		return
	}
	imageName := os.Args[1]

	model, close, err := tensor.NewModel("resnet", "myTag", "imagenet_class_index.json")
	if err != nil {
		fmt.Printf("Error loading saved model: %s\n", err.Error())
		return
	}
	defer close()

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
