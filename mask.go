package gildasai

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"math/rand"

	"github.com/fogleman/gg"
	colorful "github.com/lucasb-eyer/go-colorful"
)

type Mask struct {
	Mask  image.Image
	Box   image.Rectangle
	Score float32
	Label string
}

type MaskDetector interface {
	Detect(img image.Image) ([]Mask, error)
}

func DrawMasks(dst image.Image, masks []Mask) image.Image {
	withMasks := image.NewNRGBA(dst.Bounds())
	draw.Draw(withMasks, dst.Bounds(), dst, image.ZP, draw.Over)
	for _, m := range masks {
		unite := &uniteColor{
			b: m.Box,
			c: nextColor(),
		}
		draw.DrawMask(withMasks, m.Box, unite, m.Box.Min, m.Mask, image.ZP, draw.Over)

		label := m.Label
		score := m.Score
		draw.Draw(withMasks, dst.Bounds(),
			labelImage(fmt.Sprintf("%s (%.2f%%)", label, score), m.Box.Min, dst.Bounds()),
			image.ZP, draw.Over)
	}

	return withMasks
}

type uniteColor struct {
	b image.Rectangle
	c color.NRGBA
}

func (uc *uniteColor) ColorModel() color.Model {
	return color.NRGBAModel
}

func (uc *uniteColor) Bounds() image.Rectangle {
	return uc.b
}

func (uc *uniteColor) At(x, y int) color.Color {
	return uc.c
}

func nextColor() color.NRGBA {
	r, g, b := colorful.Hsv(float64(rand.Intn(360)), 1, 0.7).RGB255()
	return color.NRGBA{R: r, G: g, B: b, A: 128}
}

func labelImage(label string, at image.Point, bounds image.Rectangle) image.Image {
	ctx := gg.NewContext(bounds.Dx(), bounds.Dy())
	ctx.SetHexColor("#FFF")
	ctx.LoadFontFace("../static/LiberationSans-Regular.ttf", 24)
	ctx.DrawString(label, float64(at.X), float64(at.Y))
	return ctx.Image()
}
