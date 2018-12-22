package landmarks

import (
	"fmt"
	"image"
	"image/draw"
	"testing"

	"github.com/gildasch/gildas-ai/faces/detection"
	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/stretchr/testify/require"
)

func TestDetection(t *testing.T) {
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
	testImage, err := imageutils.FromFile("../pictures/2.jpg")
	require.NoError(t, err)

	detector, err := detection.NewDetectorFromFile("../detection/frozen_inference_graph_face.pb")
	require.NoError(t, err)

	dets, err := detector.Detect(testImage)
	require.NoError(t, err)

	actual := dets.Above(0.5)

	boxes := []image.Image{}
	for _, box := range actual.Boxes {
		out := image.NewRGBA(box)
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
		imageutils.AssertImageEqual(t, fmt.Sprintf("expected/2-%d-marked.png", i), withLandmarks)

		cropped := landmarks.Center(b, testImage)
		imageutils.AssertImageEqual(t, fmt.Sprintf("expected/2-%d-cropped.png", i), cropped)
	}
}
