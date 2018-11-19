package descriptors

import (
	"fmt"
	"testing"

	"github.com/gildasch/gildas-ai/image"
	"github.com/stretchr/testify/require"
)

func TestDescriptors(t *testing.T) {
	d, err := NewDescriptor()
	require.NoError(t, err)

	dd := map[int]*Descriptors{}
	for _, i := range []int{1, 2, 5, 8, 10, 11} {
		img, err := image.FromFile(fmt.Sprintf("%d.jpg", i))
		require.NoError(t, err)
		dd[i], err = d.Compute(img)
		require.NoError(t, err)
	}

	for a, da := range dd {
		for b, db := range dd {
			fmt.Printf("%d vs %d: %f\n", a, b, da.DistanceTo(db))
		}
	}
}
