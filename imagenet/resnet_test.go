package imagenet

import (
	"os"
	"testing"

	gildasai "github.com/gildasch/gildas-ai"
	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResnet(t *testing.T) {
	var modelsRoot = os.Getenv("MODELS_ROOT")
	if modelsRoot == "" {
		modelsRoot = "../"
	}
	m, close, err := NewResnet(modelsRoot)
	require.NoError(t, err)
	defer close()

	img, err := imageutils.FromFile("../testdata/casey.jpg")
	require.NoError(t, err)
	preds, err := m.Classify(img)
	require.NoError(t, err)

	expected := []gildasai.Prediction{
		{Network: "resnet_tf_1.8.0", Score: 0.4380126, Label: "wine_bottle"},
		{Network: "resnet_tf_1.8.0", Score: 0.2553244, Label: "beer_glass"},
		{Network: "resnet_tf_1.8.0", Score: 0.097641595, Label: "beer_bottle"},
		{Network: "resnet_tf_1.8.0", Score: 0.0893241, Label: "red_wine"},
		{Network: "resnet_tf_1.8.0", Score: 0.035121955, Label: "corkscrew"},
		{Network: "resnet_tf_1.8.0", Score: 0.02177835, Label: "goblet"},
		{Network: "resnet_tf_1.8.0", Score: 0.015768701, Label: "restaurant"},
		{Network: "resnet_tf_1.8.0", Score: 0.010508077, Label: "pop_bottle"},
		{Network: "resnet_tf_1.8.0", Score: 0.010461681, Label: "cocktail_shaker"},
		{Network: "resnet_tf_1.8.0", Score: 0.001482183, Label: "eggnog"}}

	assert.Equal(t, expected, preds.Best(10))
}
