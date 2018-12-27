package faceapi

import (
	"image"
	"image/draw"
	"image/png"
	"os"
	"testing"

	gildasai "github.com/gildasch/gildas-ai"
	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDetection(t *testing.T) {
	d, err := NewDetector()
	require.NoError(t, err)

	testCases := []struct {
		filename string
		expected []gildasai.Detection
	}{
		{
			filename: "pictures/1.jpg",
			expected: []gildasai.Detection{
				{Box: image.Rect(929, 316, 1000, 399), Score: 0.99272114, Class: 1},
				{Box: image.Rect(996, 276, 1074, 380), Score: 0.955024, Class: 1},
				{Box: image.Rect(825, 368, 899, 462), Score: 0.95483625, Class: 1},
				{Box: image.Rect(651, 385, 725, 469), Score: 0.93159246, Class: 1},
				{Box: image.Rect(710, 307, 780, 390), Score: 0.8155815, Class: 1}},
		},
		{
			filename: "pictures/2.jpg",
			expected: []gildasai.Detection{
				{Box: image.Rect(1084, 268, 1195, 420), Score: 0.86345625, Class: 1},
				{Box: image.Rect(823, 158, 933, 294), Score: 0.81866825, Class: 1},
				{Box: image.Rect(223, 110, 379, 301), Score: 0.7506916, Class: 1},
				{Box: image.Rect(1376, 132, 1520, 336), Score: 0.65255237, Class: 1},
				{Box: image.Rect(533, 100, 655, 283), Score: 0.6224947, Class: 1}},
		},
		{
			filename: "pictures/3.png",
			expected: []gildasai.Detection{
				{Box: image.Rect(528, 237, 762, 484), Score: 0.9831935, Class: 1}},
		},
	}

	for _, tc := range testCases {
		testImage, err := imageutils.FromFile(tc.filename)
		require.NoError(t, err)

		detections, err := d.Detect(testImage)
		require.NoError(t, err)

		actual := gildasai.Above(detections, 0.5)

		assert.Equal(t, tc.expected, actual)
	}
}

func TestDetection2(t *testing.T) {
	d, err := NewDetector()
	require.NoError(t, err)

	testImage, err := imageutils.FromFile("pictures/4.jpg")
	require.NoError(t, err)

	detections, err := d.Detect(testImage)
	require.NoError(t, err)

	actualDetections := gildasai.Above(detections, 0.5)
	require.Len(t, actualDetections, 1)

	box := actualDetections[0].Box
	actual := image.NewRGBA(box)
	draw.Draw(actual, box, testImage, box.Min, draw.Src)

	f, _ := os.Create("detection-in-4-expected.png")
	png.Encode(f, actual)
	f.Close()

	imageutils.AssertImageEqual(t, "detection-in-4-expected.png", actual)
}
