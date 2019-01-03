package gildasai

import (
	"path/filepath"

	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/pkg/errors"
)

func ExtractFacesFromFolder(path string, extractor *Extractor, store FaceStore) (current chan string, errs chan error, done chan bool, total int, err error) {
	files, err := filepath.Glob(path + "/*")
	if err != nil {
		return nil, nil, nil, 0, err
	}

	total = len(files)
	current = make(chan string)
	errs = make(chan error)
	done = make(chan bool)

	go func() {
		for _, file := range files {
			current <- file

			_, ok, _ := store.GetFaces(file)
			if ok {
				continue
			}

			img, err := imageutils.FromFile(file)
			if err != nil {
				errs <- errors.Wrapf(err, "error reading image from file %q", file)
				continue
			}

			detections, landmarks, descrs, err := extractor.ExtractPrimitives(img)
			if err != nil && err != ErrNoFaceDetected {
				errs <- errors.Wrapf(err, "error extracting face primitives from image %q", file)
				continue
			}

			if len(detections) == 0 {
				err = store.StoreFace(&FaceItem{
					Identifier: file,
					Network:    extractor.Network,
				})
				if err != nil {
					errs <- errors.Wrapf(err, "error storing face primitives from image %q", file)
				}
				continue
			}

			for i := range detections {
				err = store.StoreFace(&FaceItem{
					Identifier:  file,
					Network:     extractor.Network,
					Detection:   detections[i],
					Landmarks:   landmarks[i],
					Descriptors: descrs[i],
				})
				if err != nil {
					errs <- errors.Wrapf(err, "error storing face primitives from image %q", file)
					continue
				}
			}
		}
		done <- true
	}()

	return current, errs, done, total, nil
}
