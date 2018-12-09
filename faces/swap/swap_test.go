package swap

import (
	"image"
	"testing"

	"github.com/fogleman/gg"
	"github.com/gildasch/gildas-ai/faces"
	"github.com/gildasch/gildas-ai/faces/detection"
	"github.com/gildasch/gildas-ai/faces/landmarks"
	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/stretchr/testify/require"
)

func TestFaceSwap(t *testing.T) {
	src, err := imageutils.FromFile("1544380116.png")
	require.NoError(t, err)
	dest, err := imageutils.FromFile("20181207_182413.jpg")
	require.NoError(t, err)

	detector, err := detection.NewDetectorFromFile("../detection/frozen_inference_graph_face.pb")
	require.NoError(t, err)

	landmark, err := landmarks.NewLandmarkFromFile("../landmarks/landmarksnet", "myTag")
	require.NoError(t, err)

	extractor := &faces.Extractor{
		Detector: detector,
		Landmark: landmark,
	}

	out, err := FaceSwap(extractor, landmark, dest, src)
	require.NoError(t, err)

	gg.SavePNG("out-faceswap.png", out)
}

func TestSwap(t *testing.T) {
	src, err := imageutils.FromFile("1544380116.png")
	require.NoError(t, err)
	dest, err := imageutils.FromFile("1544394271.png")
	require.NoError(t, err)

	landmark, err := landmarks.NewLandmarkFromFile("../landmarks/landmarksnet", "myTag")
	require.NoError(t, err)

	out, err := swap(landmark, src, dest)
	require.NoError(t, err)

	gg.SavePNG("out-swap.png", out)
}

func TestMaskFromPolygon(t *testing.T) {
	in := image.NewRGBA(image.Rectangle{
		Max: image.Point{100, 100},
	})
	mask := maskFromPolygon(in, []image.Point{
		{10, 10}, {5, 65}, {17, 85}, {45, 66}, {70, 70}, {60, 30}, {50, 15},
	})

	gg.SavePNG("out.png", mask)
}

func TestMaskFromPolygonWithBounds(t *testing.T) {
	in := image.NewRGBA(image.Rectangle{
		Min: image.Point{100, 100},
		Max: image.Point{200, 200},
	})
	mask := maskFromPolygon(in, []image.Point{
		{110, 110}, {105, 165}, {117, 185}, {145, 166}, {170, 170}, {160, 130}, {150, 115},
		// {10, 10}, {5, 65}, {17, 85}, {45, 66}, {70, 70}, {60, 30}, {50, 15},
	})

	gg.SavePNG("out2.png", mask)
}
