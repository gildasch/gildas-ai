package gifutils

import (
	"fmt"
	"image"
	"image/color"
	"image/color/palette"
	"image/draw"
	"time"

	"github.com/esimov/colorquant"
)

type FloydSteinberg struct{}

var floydSteinberg = colorquant.Dither{
	[][]float32{
		[]float32{0.0, 0.0, 0.0, 7.0 / 48.0, 5.0 / 48.0},
		[]float32{3.0 / 48.0, 5.0 / 48.0, 7.0 / 48.0, 5.0 / 48.0, 3.0 / 48.0},
		[]float32{1.0 / 48.0, 3.0 / 48.0, 5.0 / 48.0, 3.0 / 48.0, 1.0 / 48.0},
	},
}

func (FloydSteinberg) Convert(src image.Image, bounds image.Rectangle, p color.Palette) *image.Paletted {
	startQuant := time.Now()
	palettedImage := floydSteinberg.Quantize(src, image.NewPaletted(bounds, palette.WebSafe), 256, true, true)
	fmt.Println("FloydSteinberg.Quantize:", time.Since(startQuant))

	return palettedImage.(*image.Paletted)
}

type StandardQuantizer struct{}

func (StandardQuantizer) Convert(src image.Image, bounds image.Rectangle, p color.Palette) *image.Paletted {
	start := time.Now()
	palettedImage := image.NewPaletted(bounds, palette.Plan9[:256])
	// palettedImage.Palette = .Quantizer.Quantize(make(color.Palette, 0, 256), m)
	draw.FloydSteinberg.Draw(palettedImage, bounds, src, image.ZP)
	fmt.Println("StandardQuantizer:", time.Since(start))

	return palettedImage
}
