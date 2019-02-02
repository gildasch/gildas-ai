package main

import (
	"fmt"
	"log"
	"os"

	"github.com/gildasch/gildas-ai/imagenet"
	"github.com/gildasch/gildas-ai/imageutils"
)

func main() {
	model, close, err := imagenet.NewPnasnet("./")
	if err != nil {
		log.Fatal(err)
	}
	defer close()

	img, err := imageutils.FromFile(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}

	preds, err := model.Classify(img)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(preds.Best(10))
}
