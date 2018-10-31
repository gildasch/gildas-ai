package image

import (
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"net/http"
	"os"

	"github.com/nfnt/resize"
	"github.com/pkg/errors"
)

func FromFile(filename string) (image.Image, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}

	img, _, err := image.Decode(f)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to decode image")
	}

	return scaled(img), nil
}

func FromURL(url string) (image.Image, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to get %q", url)
	}
	defer resp.Body.Close()

	img, _, err := image.Decode(resp.Body)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to decode image")
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
