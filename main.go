package main

import (
	"fmt"

	"github.com/gildasch/gildas-ai/image"
	"github.com/gildasch/gildas-ai/tensor"
)

func main() {
	model, close, err := tensor.NewModel("myModel", "myTag")
	if err != nil {
		fmt.Printf("Error loading saved model: %s\n", err.Error())
		return
	}
	defer close()

	imageName := "gorge2.jpg"
	img, err := image.FromFile(imageName)
	if err != nil {
		fmt.Printf("cannot read image %q: %v\n", imageName, err)
		return
	}

	preds, err := model.Inception(img)
	if err != nil {
		fmt.Printf("there was an error while running the inception: %v\n", err)
		return
	}

	best, bestV := preds.Best()
	fmt.Printf("Result value: %v (%f) \n", best, bestV)
}
