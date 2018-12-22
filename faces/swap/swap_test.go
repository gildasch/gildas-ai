package swap

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"io/ioutil"
	"testing"
	"time"

	"github.com/gildasch/gildas-ai/faces"
	"github.com/gildasch/gildas-ai/faces/detection"
	"github.com/gildasch/gildas-ai/faces/landmarks"
	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFaceSwap(t *testing.T) {
	src, err := imageutils.FromFile("gab.png")
	require.NoError(t, err)
	dest, err := imageutils.FromFile("group.jpg")
	require.NoError(t, err)

	detector, err := detection.NewDetectorFromFile("../detection/frozen_inference_graph_face.pb")
	require.NoError(t, err)

	landmark, err := landmarks.NewLandmarkFromFile("../landmarks/landmarksnet", "myTag")
	require.NoError(t, err)

	extractor := &faces.Extractor{
		Detector: detector,
		Landmark: landmark,
	}

	out, err := FaceSwap(extractor, landmark, dest, src, 0)
	require.NoError(t, err)

	assertImageEqual(t, "faceswap-expected.png", out)
}

func TestSwap(t *testing.T) {
	src, err := imageutils.FromFile("gab.png")
	require.NoError(t, err)
	dest, err := imageutils.FromFile("syl.png")
	require.NoError(t, err)

	landmark, err := landmarks.NewLandmarkFromFile("../landmarks/landmarksnet", "myTag")
	require.NoError(t, err)

	out, err := swap(landmark, src, dest, 0)
	require.NoError(t, err)

	assertImageEqual(t, "swap-expected.png", out)
}

func TestMaskFromPolygon(t *testing.T) {
	in := image.NewRGBA(image.Rectangle{
		Max: image.Point{100, 100},
	})
	mask := maskFromPolygon(in, []image.Point{
		{10, 10}, {5, 65}, {17, 85}, {45, 66}, {70, 70}, {60, 30}, {50, 15},
	})

	assertImageEqual(t, "maskFromPolygon-expected.png", mask)
}

func TestMaskFromPolygonWithBounds(t *testing.T) {
	in := image.NewRGBA(image.Rectangle{
		Min: image.Point{100, 100},
		Max: image.Point{200, 200},
	})
	mask := maskFromPolygon(in, []image.Point{
		{110, 110}, {105, 165}, {117, 185}, {145, 166}, {170, 170}, {160, 130}, {150, 115},
	})

	assertImageEqual(t, "maskFromPolygonWithBounds-expected.png", mask)
}

func assertImageEqual(t *testing.T, expectedFilename string, actual image.Image) bool {
	start := time.Now()
	defer func() {
		fmt.Println("assertImageEqual in ", time.Since(start))
	}()

	expectedBytes, err := ioutil.ReadFile(expectedFilename)
	if err != nil {
		t.Errorf("could not read expected image %q: %v", expectedFilename, err)
		t.FailNow()
		return false
	}

	var actualBytes bytes.Buffer
	err = png.Encode(&actualBytes, actual)
	if err != nil {
		t.Errorf("could not encode actual image: %v", err)
		t.FailNow()
		return false
	}

	return assert.Equal(t, expectedBytes, actualBytes.Bytes())
}
