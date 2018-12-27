package faceapi

import (
	"fmt"
	"image"
	"image/draw"
	"testing"

	gildasai "github.com/gildasch/gildas-ai"
	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/stretchr/testify/require"
)

func TestLandmarksDetection(t *testing.T) {
	l, err := NewLandmark()
	require.NoError(t, err)

	for i := 1; i < 12; i++ {
		testImage, err := imageutils.FromFile(fmt.Sprintf("testdata/single-%d.jpg", i))
		require.NoError(t, err)

		landmarks, err := l.Detect(testImage)
		require.NoError(t, err)

		withLandmarks := landmarks.DrawOnImage(testImage)
		imageutils.AssertImageEqual(t, fmt.Sprintf("expected/single-%d-marked.png", i), withLandmarks)
	}
}

func TestFullImage(t *testing.T) {
	testImage, err := imageutils.FromFile("pictures/2.jpg")
	require.NoError(t, err)

	detector, err := NewDetectorFromFile("frozen_inference_graph_face.pb")
	require.NoError(t, err)

	dets, err := detector.Detect(testImage)
	require.NoError(t, err)

	actual := gildasai.Above(dets, 0.5)

	boxes := []image.Image{}
	for _, d := range actual {
		out := image.NewRGBA(d.Box)
		draw.Draw(out, d.Box, testImage, d.Box.Min, draw.Src)
		boxes = append(boxes, out)
	}

	l, err := NewLandmark()
	require.NoError(t, err)

	for i, b := range boxes {
		fmt.Println(b.Bounds())
		landmarks, err := l.Detect(b)
		require.NoError(t, err)

		withLandmarks := landmarks.DrawOnFullImage(b, testImage)
		imageutils.AssertImageEqual(t, fmt.Sprintf("expected/2-%d-marked.png", i), withLandmarks)

		cropped := landmarks.Center(b, testImage)
		imageutils.AssertImageEqual(t, fmt.Sprintf("expected/2-%d-cropped.png", i), cropped)
	}
}
