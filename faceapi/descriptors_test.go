package faceapi

import (
	"fmt"
	"testing"

	gildasai "github.com/gildasch/gildas-ai"
	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/stretchr/testify/require"
)

func TestDescriptors(t *testing.T) {
	d, err := NewDescriptor()
	require.NoError(t, err)

	dd := map[string]gildasai.Descriptors{}
	for _, i := range []string{"1", "2", "5", "8", "10", "11", "4-face-1-cropped"} {
		img, err := imageutils.FromFile(fmt.Sprintf("%s.jpg", i))
		require.NoError(t, err)
		dd[i], err = d.Compute(img)
		require.NoError(t, err)
	}

	for a, da := range dd {
		for b, db := range dd {
			dist, err := da.DistanceTo(db)
			fmt.Printf("%s vs %s: %f / error %v\n", a, b, dist, err)
		}
	}
}
