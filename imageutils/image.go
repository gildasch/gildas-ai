package imageutils

import (
	"archive/zip"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
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

	return img, nil
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

	return img, nil

}

func Scaled(img image.Image, height, width uint) image.Image {
	return resize.Resize(width, height, img, resize.NearestNeighbor)
}

func FromZip(zipFile io.ReaderAt, size int64) (images map[string]image.Image, errs []error) {
	r, err := zip.NewReader(zipFile, size)
	if err != nil {
		return nil, []error{errors.Wrap(err, "error opening zip file")}
	}

	images = map[string]image.Image{}
	for _, f := range r.File {
		rc, err := f.Open()
		if err != nil {
			errs = append(errs,
				errors.Wrapf(err, "error opening file %s of zip file", f.FileHeader.Name))
			continue
		}

		img, _, err := image.Decode(rc)
		rc.Close()
		if err == image.ErrFormat { // not an image
			continue
		}
		if err != nil {
			errs = append(errs,
				errors.Wrapf(err, "error decoding image %s of zip file", f.FileHeader.Name))
			continue
		}

		images[f.FileHeader.Name] = img
	}

	return
}
