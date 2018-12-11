package gifutils

import (
	"image"
	"image/color"
	"image/gif"
	"time"
)

type Converter interface {
	Convert(src image.Image, bounds image.Rectangle, p color.Palette) *image.Paletted
}

const (
	defaultWidth  = 240
	defaultHeight = 240
)

func MakeGIFFromImages(in []image.Image, delay time.Duration, converter Converter) (*gif.GIF, error) {
	outGif := &gif.GIF{}
	for _, i := range in {
		// Add new frame to animated GIF
		if paletted, ok := i.(*image.Paletted); ok {
			outGif.Image = append(outGif.Image, paletted)
		} else {
			outGif.Image = append(outGif.Image, converter.Convert(i, i.Bounds(), nil))
		}
		outGif.Delay = append(outGif.Delay, int(delay.Seconds()*100)) // delay is in 100th of second
	}

	return outGif, nil
}
