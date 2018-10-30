package image

import (
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"os"

	"github.com/nfnt/resize"
)

func FromFile(filename string) (image.Image, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}

	img, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}

	return scaled(img), nil
}

const (
	width  = 299
	height = 299
)

func scaled(img image.Image) image.Image {
	return resize.Resize(width, height, img, resize.NearestNeighbor)
}
