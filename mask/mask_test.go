package mask

import (
	"fmt"
	"image"
	"image/draw"
	"testing"

	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunRCNN(t *testing.T) {
	r, err := NewRCNN("mask_rcnn_coco", "myTag")
	require.NoError(t, err)

	img, err := imageutils.FromFile("testdata/ruedesmartyrs.jpg")
	require.NoError(t, err)

	detections, masks, err := r.Detect(img)
	require.NoError(t, err)

	maskImages := masks.GetAllOnImage(detections, img)

	withMasks := image.NewNRGBA(img.Bounds())
	draw.Draw(withMasks, img.Bounds(), img, image.ZP, draw.Over)

	for _, mi := range maskImages {
		draw.Draw(withMasks, img.Bounds(), mi, image.ZP, draw.Over)
	}

	imageutils.AssertImageEqual(t, "testdata/ruedesmartyrs-expected.png", withMasks)
}

func TestGenerateAnchors(t *testing.T) {
	actual := generateAnchors(32, []float32{0.5, 1, 2}, []float32{256, 256}, 4, 1)

	fmt.Println(actual[:9])
	fmt.Println(actual[12345])
	fmt.Println(actual[123456])
	fmt.Println(actual[196605:])

	inEpsilon(t, [][4]float32{
		{-22.6274, -11.3137, 22.6274, 11.3137},
		{-16, -16, 16, 16},
		{-11.3137, -22.6274, 11.3137, 22.6274},
		{-22.6274, -7.3137, 22.6274, 15.3137},
		{-16, -12, 16, 20},
		{-11.3137, -18.6274, 11.3137, 26.6274},
		{-22.6274, -3.3137, 22.6274, 19.3137},
		{-16, -8, 16, 24},
		{-11.3137, -14.6274, 11.3137, 30.6274},
	}, actual[:9], 0.0001)
	inEpsilon(t, [][4]float32{{41.3725, 64.6862, 86.6274, 87.3137}}, [][4]float32{actual[12345]}, 0.0001)
	inEpsilon(t, [][4]float32{{617.3725, 756.6862, 662.6274, 779.3137}}, [][4]float32{actual[123456]}, 0.0001)
	inEpsilon(t, [][4]float32{
		{997.3725, 1008.6862, 1042.6274, 1031.3137},
		{1004, 1004, 1036, 1036},
		{1008.6862, 997.3725, 1031.3137, 1042.6274},
	}, actual[196605:], 0.0001)
}

func inEpsilon(t *testing.T, expected, actual [][4]float32, epsilon float64) {
	for i := range expected {
		for j := range expected[i] {
			assert.InEpsilon(t, expected[i][j], actual[i][j], epsilon)
		}
	}
}
