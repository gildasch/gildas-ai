package imageutils

import (
	"archive/zip"
	"bytes"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"io/ioutil"
	"net/http"
	"os"

	"github.com/disintegration/imaging"
	"github.com/nfnt/resize"
	"github.com/pkg/errors"
	"github.com/rwcarlsen/goexif/exif"
)

func FromFile(filename string) (image.Image, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}

	img, _, err := Decode(f)
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

	img, _, err := Decode(resp.Body)
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

		img, _, err := Decode(rc)
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

// taken from https://github.com/edwvee/exiffix
func Decode(r io.Reader) (image.Image, string, error) {
	b, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, "", err
	}

	img, fmt, err := image.Decode(bytes.NewBuffer(b))
	if err != nil {
		return img, fmt, err
	}
	orientation := getOrientation(bytes.NewBuffer(b))
	switch orientation {
	case "1":
	case "2":
		img = imaging.FlipV(img)
	case "3":
		img = imaging.Rotate180(img)
	case "4":
		img = imaging.Rotate180(imaging.FlipV(img))
	case "5":
		img = imaging.Rotate270(imaging.FlipV(img))
	case "6":
		img = imaging.Rotate270(img)
	case "7":
		img = imaging.Rotate90(imaging.FlipV(img))
	case "8":
		img = imaging.Rotate90(img)
	}

	return img, fmt, err
}

func getOrientation(r io.Reader) string {
	x, err := exif.Decode(r)
	if err != nil {
		return "1"
	}
	if x != nil {
		orient, err := x.Get(exif.Orientation)
		if err != nil {
			return "1"
		}
		if orient != nil {
			return orient.String()
		}
	}

	return "1"
}
