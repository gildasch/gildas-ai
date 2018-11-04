package main

import (
	"fmt"
	goimage "image"
	"os"
	"strings"

	"github.com/gildasch/gildas-ai/api"
	"github.com/gildasch/gildas-ai/image"
	"github.com/gildasch/gildas-ai/tensor"
	"github.com/gin-gonic/gin"
)

func usage() {
	fmt.Printf("Usage: %s [xception|resnet] path/to/image.jpg\n", os.Args[0])
	fmt.Printf("Usage: %s web\n", os.Args[0])
}

func main() {
	models := map[string]*tensor.Model{
		"xception": &tensor.Model{
			ModelName:   "myModel",
			TagName:     "myTag",
			InputLayer:  "input_1",
			OutputLayer: "predictions/Softmax",
			ImageMode:   tensor.ImageModeTensorflow,
			Labels:      "imagenet_class_index.json",
			ImageHeight: 299,
			ImageWidth:  299,
		},
		"resnet": &tensor.Model{
			ModelName:   "resnet",
			TagName:     "myTag",
			InputLayer:  "input_1",
			OutputLayer: "fc1000/Softmax",
			ImageMode:   tensor.ImageModeCaffe,
			Labels:      "imagenet_class_index.json",
			ImageHeight: 224,
			ImageWidth:  224,
		},
		"nasnet": &tensor.Model{
			ModelName:   "nasnet-mobile",
			TagName:     "myTag",
			InputLayer:  "input_1",
			OutputLayer: "predictions/Softmax",
			ImageMode:   tensor.ImageModeTensorflow,
			Labels:      "imagenet_class_index.json",
			ImageHeight: 224,
			ImageWidth:  224,
		},
	}

	if len(os.Args) >= 2 && os.Args[1] == "web" {
		classifiers := map[string]api.Classifier{}
		for name, m := range models {
			close, err := m.Load()
			if err != nil {
				fmt.Printf("Error loading saved model: %s\n", err.Error())
				continue
			}
			defer close()

			classifiers[name] = m
		}

		app := gin.Default()
		app.LoadHTMLFiles("templates/predictions.html")
		app.GET("api", api.ClassifyHandler(classifiers, false))
		app.GET("/", api.ClassifyHandler(classifiers, true))

		app.Run()
	}

	if len(os.Args) < 3 {
		usage()
		return
	}
	modelName := os.Args[1]
	imageName := os.Args[2]

	model, ok := models[modelName]
	if !ok {
		usage()
		return
	}
	close, err := model.Load()
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
