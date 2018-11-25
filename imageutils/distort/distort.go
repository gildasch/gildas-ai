package distort

import (
	"bytes"
	"image"
	"image/png"

	"github.com/pkg/errors"
	"gopkg.in/gographics/imagick.v3/imagick"
)

func Distort(img image.Image, src []image.Point, dst []image.Point) (image.Image, error) {
	params, err := toDistortParams(src, dst)
	if err != nil {
		return nil, err
	}

	imagick.Initialize()
	defer imagick.Terminate()

	wand := imagick.NewMagickWand()

	blob, err := toBlob(img)
	if err != nil {
		return nil, err
	}

	if err := wand.ReadImageBlob(blob); err != nil {
		return nil, err
	}

	if err := wand.DistortImage(imagick.DISTORTION_SHEPARDS, params, false); err != nil {
		return nil, err
	}

	wand.SetImageFormat("PNG")

	out, err := fromBlob(wand.GetImageBlob())
	if err != nil {
		return nil, err
	}

	return out, nil
}

func toDistortParams(src []image.Point, dst []image.Point) ([]float64, error) {
	if len(src) != len(dst) {
		return nil, errors.Errorf("src length %d and dst length %d don't match", len(src), len(dst))
	}

	params := []float64{}
	for i := 0; i < len(src); i++ {
		params = append(params, []float64{
			float64(src[i].X), float64(src[i].Y), float64(dst[i].X), float64(dst[i].Y),
		}...)
	}

	return params, nil
}

func toBlob(img image.Image) ([]byte, error) {
	var buff bytes.Buffer
	err := png.Encode(&buff, img)
	if err != nil {
		return nil, err
	}

	return buff.Bytes(), nil
}

func fromBlob(blob []byte) (image.Image, error) {
	return png.Decode(bytes.NewBuffer(blob))
}
