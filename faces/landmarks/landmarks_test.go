package landmarks

import (
	"fmt"
	goimage "image"
	"image/jpeg"
	"os"
	"testing"

	"github.com/gildasch/gildas-ai/image"
	"github.com/stretchr/testify/require"
)

func TestDetection(t *testing.T) {
	l, err := NewLandmark()
	require.NoError(t, err)

	for i := 1; i < 12; i++ {
		testImage, err := image.FromFile(fmt.Sprintf("%d.jpg", i))
		require.NoError(t, err)

		landmarks, err := l.Detect(testImage)
		require.NoError(t, err)

		saveLandmarksImage(testImage, landmarks, fmt.Sprintf("%d-marked.jpg", i))
	}
}

func saveLandmarksImage(img goimage.Image, landmarks *Landmarks, filename string) {
	withLandmarks := landmarks.DrawOnImage(img)

	outf, _ := os.Create(filename)
	jpeg.Encode(outf, withLandmarks, nil)
}

func TestGenerateData(t *testing.T) {
	l, err := NewLandmark()
	require.NoError(t, err)

	for _, i := range []string{"a", "b", "c", "d"} {
		testImage, err := image.FromFile(fmt.Sprintf("%s.png", i))
		require.NoError(t, err)

		landmarks, err := l.Detect(testImage)
		require.NoError(t, err)

		cropped := landmarks.Center(testImage)
		outf, _ := os.Create(fmt.Sprintf("%s-cropped.jpg", i))
		jpeg.Encode(outf, cropped, nil)
	}
}

func TestGenerateData2(t *testing.T) {
	l, err := NewLandmark()
	require.NoError(t, err)

	for i := 1; i <= 11; i++ {
		testImage, err := image.FromFile(fmt.Sprintf("%d.jpg", i))
		require.NoError(t, err)

		landmarks, err := l.Detect(testImage)
		require.NoError(t, err)

		cropped := landmarks.Center(testImage)
		outf, _ := os.Create(fmt.Sprintf("%d-cropped.jpg", i))
		jpeg.Encode(outf, cropped, nil)
	}
}
