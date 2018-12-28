package sqlite

import (
	"os"
	"testing"

	gildasai "github.com/gildasch/gildas-ai"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStoreAndGet(t *testing.T) {
	s, err := NewStore("/tmp/gildasai.test.sqlite")
	require.NoError(t, err)
	defer s.Close()
	defer os.Remove("/tmp/gildasai.test.sqlite")

	for _, item := range testPredictionItems {
		err = s.StorePrediction(item.Identifier, item)
		require.NoError(t, err)
	}

	actual, ok, err := s.GetPrediction("a journey to the stars")
	require.NoError(t, err)
	assert.True(t, ok)
	assert.Equal(t, testPredictionItems[1], actual)

	actual, ok, err = s.GetPrediction("a trip to the moon")
	require.NoError(t, err)
	assert.False(t, ok)
	assert.Nil(t, actual)
}

func TestStoreAndSearch(t *testing.T) {
	s, err := NewStore("/tmp/gildasai.test.sqlite")
	require.NoError(t, err)
	defer s.Close()
	defer os.Remove("/tmp/gildasai.test.sqlite")

	for _, item := range testPredictionItems {
		err = s.StorePrediction(item.Identifier, item)
		require.NoError(t, err)
	}

	actuals, err := s.SearchPrediction("keyboard", "", 10)
	require.NoError(t, err)
	if assert.Len(t, actuals, 1) {
		assert.Equal(t, testPredictionItems[2], actuals[0])
	}

	actuals, err = s.SearchPrediction("mouse", "", 10)
	require.NoError(t, err)
	assert.Len(t, actuals, 0)
}

var testPredictionItems = []*gildasai.PredictionItem{
	{
		Identifier: "a wonderful picture of my ox",
		Predictions: gildasai.Predictions{
			{Network: "ImageFaceNet C-CNN", Score: 0.9999, Label: "ox"},
			{Network: "ImageFaceNet C-CNN", Score: 0.7658, Label: "wonderful"},
			{Network: "ImageFaceNet C-CNN", Score: 0.01233564, Label: "dog"},
			{Network: "ImageFaceNet C-CNN", Score: 0.0001, Label: "personal_computer"},
		},
	},
	{
		Identifier: "a journey to the stars",
		Predictions: gildasai.Predictions{
			{Network: "ImageFaceNet C-CNN", Score: 0.8593, Label: "night"},
			{Network: "ImageFaceNet C-CNN", Score: 0.0345, Label: "alps"},
			{Network: "ImageFaceNet C-CNN", Score: 0.011, Label: "fork"},
		},
	},
	{
		Identifier: "my desk",
		Predictions: gildasai.Predictions{
			{Network: "ImageFaceNet C-CNN", Score: 0.769, Label: "tv"},
			{Network: "ImageFaceNet C-CNN", Score: 0.4789, Label: "computer_keyboard"},
			{Network: "ImageFaceNet C-CNN", Score: 0.1621, Label: "typewriter_keyboard"},
			{Network: "ImageFaceNet C-CNN", Score: 0.011, Label: "fork"},
		},
	},
}
