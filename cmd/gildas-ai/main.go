package main

import (
	"fmt"
	"image"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	gildasai "github.com/gildasch/gildas-ai"
	"github.com/gildasch/gildas-ai/api"
	"github.com/gildasch/gildas-ai/faceapi"
	"github.com/gildasch/gildas-ai/imagenet"
	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/gildasch/gildas-ai/maskrcnn"
	"github.com/gildasch/gildas-ai/sqlite"
	"github.com/gin-contrib/cache"
	"github.com/gin-contrib/cache/persistence"
	"github.com/gin-gonic/gin"
)

func usage() {
	fmt.Printf("Usage: %s [xception|resnet] path/to/image.jpg\n", os.Args[0])
	fmt.Printf("Usage: %s web\n", os.Args[0])
}

func main() {
	modelsRoot := os.Getenv("MODELS_ROOT")

	models := map[string]func(modelRoot string) (*imagenet.Model, func() error, error){
		"xception": imagenet.NewXception,
		"resnet":   imagenet.NewResnet,
		"nasnet":   imagenet.NewNasnet,
		"pnasnet":  imagenet.NewPnasnet,
	}

	detector, err := faceapi.NewDetectorFromFile("faceapi/frozen_inference_graph_face.pb")
	if err != nil {
		log.Fatal(err)
	}

	landmark, err := faceapi.NewLandmarkFromFile(
		modelsRoot+"models/face-api-js-landmarks/face-api-landmarksnet_tf_1.8.0", "myTag")
	if err != nil {
		log.Fatal(err)
	}

	descriptor, err := faceapi.NewDescriptorFromFile(
		modelsRoot+"models/face-api-js-descriptors/face-api-descriptors_tf_1.8.0", "myTag")
	if err != nil {
		log.Fatal(err)
	}

	if len(os.Args) >= 2 && os.Args[1] == "web" {
		classifiers := map[string]gildasai.Classifier{}
		for name, load := range models {
			m, close, err := load(modelsRoot)
			if err != nil {
				fmt.Printf("Error loading saved model: %s\n", err.Error())
				continue
			}
			defer close()

			classifiers[name] = m
		}

		sqliteLocation := ".inception.sqlite"
		if os.Getenv("SQLITE_STORE") != "" {
			sqliteLocation = os.Getenv("SQLITE_STORE")
		}
		sqliteStore, err := sqlite.NewStore(sqliteLocation)
		if err != nil {
			log.Fatal(err)
		}

		extractor := &gildasai.Extractor{
			Detector:   detector,
			Landmark:   landmark,
			Descriptor: descriptor}

		app := gin.Default()
		app.Static("/static", "./static")
		app.LoadHTMLFiles(
			"templates/index.html",
			"templates/predictions.html",
			"templates/faces.html",
			"templates/photos.html",
			"templates/faceswap.html",
			"templates/masks.html",
			"templates/facesearch.html")
		app.GET("/", func(c *gin.Context) {
			c.HTML(http.StatusOK, "index.html", nil)
		})
		app.GET("/object/api", api.ClassifyHandler(classifiers, false))
		app.GET("/object", api.ClassifyHandler(classifiers, true))

		batches := map[string]*gildasai.Batch{}
		app.GET("/faces", api.FacesHomeHandler(batches))
		app.POST("/faces", api.FacesPostBatchHandler(extractor, batches))
		app.GET("/faces/batch/:batchID", api.FacesGetBatchHandler(batches))
		app.GET("/faces/batch/:batchID/sources/:name", api.FaceSourceHandler(batches))
		app.GET("/faces/batch/:batchID/cropped/:name", api.FaceCroppedHandler(batches))

		app.GET("/photos", api.PhotosHandler(sqliteStore))
		app.GET("/photos/*filename", api.GetPhotoHandler(sqliteStore))

		store := persistence.NewInMemoryStore(365 * 24 * time.Hour)
		app.GET("/faceswap", cache.CachePage(store, 12*time.Hour, api.FaceSwapHandler(extractor, landmark)))

		if modelsRoot != "" {
			modelsRoot += "mask/"
		} else {
			modelsRoot = "maskrcnn/"
		}
		maskDetector, err := maskrcnn.NewRCNN(modelsRoot+"mask_rcnn_coco_tf_1.8.0", "myTag")
		if err != nil {
			log.Fatal(err)
		}
		masksStore := map[string]api.MaskResult{}
		app.GET("/masks", api.MaskHandler(maskDetector, masksStore))
		app.GET("/masks/result.jpg", api.MaskImageHandler(masksStore))

		clusters, err := api.CalculateClusters(sqliteStore)
		if err != nil {
			log.Fatal(err)
		}

		app.GET("/facesearch", api.FacesearchHandler(sqliteStore, clusters))
		app.GET("/facesearch/:detection/matches", api.FacesearchDetectionHandler(sqliteStore, clusters))
		app.GET("/facesearch/:detection/against/:detection2", api.FacesearchAgainstHandler(sqliteStore))
		app.GET("/facesearch/:detection/detection.jpg", api.FacesearchDetectionImageHandler())
		app.GET("/facesearch/:detection/landmarks.jpg", api.FacesearchLandmarkImageHandler(sqliteStore))

		app.Run()
	}

	if len(os.Args) < 3 {
		usage()
		return
	}
	modelName := os.Args[1]
	imageName := os.Args[2]

	load, ok := models[modelName]
	if !ok {
		usage()
		return
	}
	model, close, err := load(modelsRoot)
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

	preds, err := model.Classify(img)
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
