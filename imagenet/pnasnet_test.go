package imagenet

import (
	"os"
	"testing"

	gildasai "github.com/gildasch/gildas-ai"
	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPnasnet(t *testing.T) {
	var modelsRoot = os.Getenv("MODELS_ROOT")
	if modelsRoot == "" {
		modelsRoot = "../"
	}
	m, close, err := NewPnasnet(modelsRoot)
	require.NoError(t, err)
	defer close()

	img, err := imageutils.FromFile("../testdata/casey.jpg")
	require.NoError(t, err)
	preds, err := m.Classify(img)
	require.NoError(t, err)

	expected := []gildasai.Prediction{
		{Network: "pnasnet_tf_" + tfVersion(), Score: 0.3297706, Label: "red_wine"},
		{Network: "pnasnet_tf_" + tfVersion(), Score: 0.22555843, Label: "wine_bottle"},
		{Network: "pnasnet_tf_" + tfVersion(), Score: 0.10641288, Label: "beer_glass"},
		{Network: "pnasnet_tf_" + tfVersion(), Score: 0.08965542, Label: "goblet"},
		{Network: "pnasnet_tf_" + tfVersion(), Score: 0.010835887, Label: "beer_bottle"},
		{Network: "pnasnet_tf_" + tfVersion(), Score: 0.0077211545, Label: "restaurant"},
		{Network: "pnasnet_tf_" + tfVersion(), Score: 0.004854392, Label: "corkscrew"},
		{Network: "pnasnet_tf_" + tfVersion(), Score: 0.0019973805, Label: "water_jug"},
		{Network: "pnasnet_tf_" + tfVersion(), Score: 0.0015654737, Label: "cocktail_shaker"},
		{Network: "pnasnet_tf_" + tfVersion(), Score: 0.0014123259, Label: "pop_bottle"}}

	assert.Equal(t, expected, preds.Best(10))
}
