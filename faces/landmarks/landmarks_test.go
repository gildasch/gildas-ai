package landmarks

import (
	"fmt"
	goimage "image"
	"image/draw"
	"image/jpeg"
	"os"
	"testing"

	"github.com/gildasch/gildas-ai/faces/detection"
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

		cropped := landmarks.Center(testImage, testImage)
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

		cropped := landmarks.Center(testImage, testImage)
		outf, _ := os.Create(fmt.Sprintf("%d-cropped.jpg", i))
		jpeg.Encode(outf, cropped, nil)
	}
}

func TestGenerateData3(t *testing.T) {
	l, err := NewLandmark()
	require.NoError(t, err)

	testImage, err := image.FromFile("4-face-1.jpg")
	require.NoError(t, err)

	landmarks, err := l.Detect(testImage)
	require.NoError(t, err)

	cropped := landmarks.Center(testImage, testImage)
	outf, _ := os.Create("4-face-1-cropped.jpg")
	jpeg.Encode(outf, cropped, nil)
}

func TestFullImage(t *testing.T) {
	testImage, err := image.FromFile("../pictures/2.jpg")
	require.NoError(t, err)

	detector, err := detection.NewDetectorFromFile("../detection/frozen_inference_graph_face.pb")
	require.NoError(t, err)

	dets, err := detector.Detect(testImage)
	require.NoError(t, err)

	actual := dets.Above(0.5)

	boxes := []goimage.Image{}
	for _, box := range actual.Boxes {
		out := goimage.NewRGBA(box)
		draw.Draw(out, box, testImage, box.Min, draw.Src)
		boxes = append(boxes, out)
	}

	l, err := NewLandmark()
	require.NoError(t, err)

	for i, b := range boxes {
		fmt.Println(b.Bounds())
		landmarks, err := l.Detect(b)
		require.NoError(t, err)

		withLandmarks := landmarks.DrawOnFullImage(b, testImage)

		outf, _ := os.Create(fmt.Sprintf("2-%d-marked.jpg", i))
		jpeg.Encode(outf, withLandmarks, nil)

		cropped := landmarks.Center(b, testImage)
		outf2, _ := os.Create(fmt.Sprintf("2-%d-cropped.jpg", i))
		jpeg.Encode(outf2, cropped, nil)
	}
}
