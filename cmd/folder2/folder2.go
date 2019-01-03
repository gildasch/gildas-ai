package main

import (
	"fmt"
	"log"
	"os"
	"strings"

	gildasai "github.com/gildasch/gildas-ai"
	"github.com/gildasch/gildas-ai/faceapi"
	"github.com/gildasch/gildas-ai/sqlite"
)

func usage() {
	fmt.Printf("%s [model-root-folder] [image-folder]\n", os.Args[0])
}

func main() {
	if len(os.Args) < 3 {
		usage()
		return
	}

	modelRootFolder := strings.TrimSuffix(os.Args[1], "/")
	imageFolder := strings.TrimSuffix(os.Args[2], "/")

	extractor, err := faceapi.NewDefaultExtractor(modelRootFolder)
	if err != nil {
		log.Fatal("could not load face extractor: ", err)
	}

	store, err := sqlite.NewStore(imageFolder + "/.inception.sqlite")
	if err != nil {
		log.Fatal("could not create store with file "+imageFolder+"/.inception.sqlite: ", err)
	}
	defer store.Close()

	current, errors, done, total, err := gildasai.ExtractFacesFromFolder(imageFolder, extractor, store)
	if err != nil {
		log.Fatal("could not run the extraction: ", err)
	}

	var c string
	processed := 0
	for {
		fmt.Printf("\rprogress: %d/%d (current: %s)", processed, total, c)
		select {
		case c = <-current:
		case err = <-errors:
			fmt.Printf("\nerror on file %d/%q: %v\n", processed, c, err)
		case <-done:
			fmt.Printf("\nthe extraction in complete\n")
			return
		}
		processed++
	}
}
