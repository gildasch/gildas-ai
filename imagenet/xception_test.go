package imagenet

import (
	"os"
	"testing"

	gildasai "github.com/gildasch/gildas-ai"
	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestXception(t *testing.T) {
	var modelsRoot = os.Getenv("MODELS_ROOT")
	if modelsRoot == "" {
		modelsRoot = "../"
	}
	m, close, err := NewXception(modelsRoot)
	require.NoError(t, err)
	defer close()

	img, err := imageutils.FromFile("../testdata/casey.jpg")
	require.NoError(t, err)
	preds, err := m.Classify(img)
	require.NoError(t, err)

	expected := []gildasai.Prediction{
		{Network: "xception_tf_" + tfVersion(), Score: 0.0010001443, Label: "rain_barrel"},
		{Network: "xception_tf_" + tfVersion(), Score: 0.00100012, Label: "folding_chair"},
		{Network: "xception_tf_" + tfVersion(), Score: 0.0010001108, Label: "rotisserie"},
		{Network: "xception_tf_" + tfVersion(), Score: 0.0010001083, Label: "bikini"},
		{Network: "xception_tf_" + tfVersion(), Score: 0.001000094, Label: "dishrag"},
		{Network: "xception_tf_" + tfVersion(), Score: 0.0010000922, Label: "cellular_telephone"},
		{Network: "xception_tf_" + tfVersion(), Score: 0.0010000867, Label: "speedboat"},
		{Network: "xception_tf_" + tfVersion(), Score: 0.0010000864, Label: "washer"},
		{Network: "xception_tf_" + tfVersion(), Score: 0.0010000862, Label: "remote_control"},
		{Network: "xception_tf_" + tfVersion(), Score: 0.0010000856, Label: "gondola"}}

	assert.Equal(t, expected, preds.Best(10))
}
