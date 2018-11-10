package detection

import (
	goimage "image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"os"
	"testing"

	"github.com/gildasch/gildas-ai/image"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDetection(t *testing.T) {
	d, err := NewDetector()
	require.NoError(t, err)

	testCases := []struct {
		filename              string
		expectedNumDetections int
	}{
		{
			filename:              "../pictures/1.jpg",
			expectedNumDetections: 5,
		},
		{
			filename:              "../pictures/2.jpg",
			expectedNumDetections: 5,
		},
		{
			filename:              "../pictures/3.png",
			expectedNumDetections: 1,
		},
	}

	for _, tc := range testCases {
		testImage, err := image.FromFile(tc.filename)
		require.NoError(t, err)

		detections, err := d.Detect(testImage)
		require.NoError(t, err)

		actual := detections.Above(0.5)

		assert.Len(t, actual.Boxes, tc.expectedNumDetections)
		assert.Len(t, actual.Scores, tc.expectedNumDetections)
		assert.Len(t, actual.Classes, tc.expectedNumDetections)
		assert.Equal(t, tc.expectedNumDetections, actual.NumDetections)
	}
}

func saveImage(img goimage.Image, detections Detections, filename string) {
	out := goimage.NewRGBA(img.Bounds())
	draw.Draw(out, out.Bounds(), img, goimage.ZP, draw.Src)

	for _, box := range detections.Boxes {
		colorRectangle(out, box.Min.X, box.Min.Y, box.Max.X, box.Max.Y)
	}

	outf, _ := os.Create(filename)
	jpeg.Encode(outf, out, nil)
}

func colorRectangle(img *goimage.RGBA, x1, y1, x2, y2 int) {
	hLine(img, x1, x2, y1)
	hLine(img, x1, x2, y2)
	vLine(img, x1, y1, y2)
	vLine(img, x2, y1, y2)
}

func hLine(img *goimage.RGBA, x1, x2, y int) {
	for i := x1; i < x2; i++ {
		img.Set(i, y, color.RGBA{G: 255})
	}
}

func vLine(img *goimage.RGBA, x, y1, y2 int) {
	for j := y1; j < y2; j++ {
		img.Set(x, j, color.RGBA{G: 255})
	}
}
