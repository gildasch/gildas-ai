package mask

import (
	"fmt"
	"testing"

	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/stretchr/testify/require"
)

func TestNewRCNN(t *testing.T) {
	r, err := NewRCNN("mask_rcnn_coco", "myTag")
	require.NoError(t, err)

	fmt.Println(r)
}

func TestRunRCNN(t *testing.T) {
	r, err := NewRCNN("mask_rcnn_coco", "myTag")
	require.NoError(t, err)

	img, err := imageutils.FromURL("https://upload.wikimedia.org/wikipedia/commons/8/82/Denmark_Street_in_2010%2C_viewed_from_its_junction_with_Charing_Cross_Road%2C_by_David_Dixon%2C_geograph.org.uk_1665474.jpg")
	require.NoError(t, err)

	s, err := r.Inception(img)
	require.NoError(t, err)

	fmt.Println(s)
}
