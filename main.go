package main

import (
	"fmt"
	"image"
	"log"
	"os"
	"strings"

	"github.com/gildasch/gildas-ai/api"
	"github.com/gildasch/gildas-ai/faces"
	"github.com/gildasch/gildas-ai/faces/descriptors/faceapi"
	"github.com/gildasch/gildas-ai/faces/detection"
	"github.com/gildasch/gildas-ai/faces/landmarks"
	"github.com/gildasch/gildas-ai/imageutils"
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
		"pnasnet": &tensor.Model{
			ModelName:       "pnasnet",
			TagName:         "myTag",
			InputLayer:      "module/hub_input/images",
			OutputLayer:     "module/final_layer/predictions",
			ImageMode:       tensor.ImageModeTensorflowPositive,
			Labels:          "imagenet_class_index.json",
			ImageHeight:     331,
			ImageWidth:      331,
			IndexCorrection: -1,
		},
	}

	detector, err := detection.NewDetectorFromFile("faces/detection/frozen_inference_graph_face.pb")
	if err != nil {
		log.Fatal(err)
	}

	landmark, err := landmarks.NewLandmarkFromFile("faces/landmarks/landmarksnet", "myTag")
	if err != nil {
		log.Fatal(err)
	}

	descriptor, err := faceapi.NewDescriptorFromFile("faces/descriptors/faceapi/descriptorsnet", "myTag")
	if err != nil {
		log.Fatal(err)
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
		app.LoadHTMLFiles("templates/predictions.html", "templates/faces.html")
		app.GET("/object/api", api.ClassifyHandler(classifiers, false))
		app.GET("/object", api.ClassifyHandler(classifiers, true))

		batches := map[string]*faces.Batch{}
		app.GET("/faces", api.FacesHomeHandler(batches))
		app.POST("/faces", api.FacesPostBatchHandler(&faces.Extractor{
			Detector:   detector,
			Landmark:   landmark,
			Descriptor: descriptor}, batches))
		app.GET("/faces/batch/:batchID", api.FacesGetBatchHandler(batches))
		app.GET("/faces/batch/:batchID/sources/:name", api.FaceSourceHandler(batches))
		app.GET("/faces/batch/:batchID/cropped/:name", api.FaceCroppedHandler(batches))

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

	var img image.Image
	if strings.HasPrefix(imageName, "https://") || strings.HasPrefix(imageName, "http://") {
		img, err = imageutils.FromURL(imageName)
		if err != nil {
			fmt.Printf("cannot read remote image %q: %v\n", imageName, err)
			return
		}
	} else {
		img, err = imageutils.FromFile(imageName)
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
