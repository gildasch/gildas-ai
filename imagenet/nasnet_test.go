package imagenet

import (
	"os"
	"testing"

	gildasai "github.com/gildasch/gildas-ai"
	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNasnet(t *testing.T) {
	var modelsRoot = os.Getenv("MODELS_ROOT")
	if modelsRoot == "" {
		modelsRoot = "../"
	}
	m, close, err := NewNasnet(modelsRoot)
	require.NoError(t, err)
	defer close()

	img, err := imageutils.FromFile("../testdata/casey.jpg")
	require.NoError(t, err)
	preds, err := m.Classify(img)
	require.NoError(t, err)

	expected := []gildasai.Prediction{
		{Network: "nasnet-mobile_tf_1.8.0", Score: 0.3846703, Label: "wine_bottle"},
		{Network: "nasnet-mobile_tf_1.8.0", Score: 0.27926758, Label: "red_wine"},
		{Network: "nasnet-mobile_tf_1.8.0", Score: 0.06262191, Label: "goblet"},
		{Network: "nasnet-mobile_tf_1.8.0", Score: 0.039620694, Label: "beer_bottle"},
		{Network: "nasnet-mobile_tf_1.8.0", Score: 0.029348562, Label: "restaurant"},
		{Network: "nasnet-mobile_tf_1.8.0", Score: 0.023344118, Label: "beer_glass"},
		{Network: "nasnet-mobile_tf_1.8.0", Score: 0.010598952, Label: "cocktail_shaker"},
		{Network: "nasnet-mobile_tf_1.8.0", Score: 0.008948702, Label: "corkscrew"},
		{Network: "nasnet-mobile_tf_1.8.0", Score: 0.008738155, Label: "dining_table"},
		{Network: "nasnet-mobile_tf_1.8.0", Score: 0.008304786, Label: "eggnog"}}

	assert.Equal(t, expected, preds.Best(10))
}
