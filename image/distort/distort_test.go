package distort

import (
	"image"
	"image/jpeg"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDistort(t *testing.T) {
	f, err := os.Open("1.jpg")
	require.NoError(t, err)

	img, err := jpeg.Decode(f)
	require.NoError(t, err)

	img, err = Distort(img, []image.Point{{
		X: 10,
		Y: 10,
	}, {
		X: 61,
		Y: 73,
	}}, []image.Point{{
		X: 0,
		Y: 0,
	}, {
		X: 71,
		Y: 83,
	}})
	require.NoError(t, err)

	fout, err := os.Create("1_out.jpg")
	require.NoError(t, err)

	err = jpeg.Encode(fout, img, nil)
	require.NoError(t, err)
}
