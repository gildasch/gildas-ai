package sqlite

import (
	"os"
	"testing"

	"github.com/gildasch/gildas-ai/tensor"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const tempSQLiteFile = "/tmp/test.sqlite"

func TestCache(t *testing.T) {
	c, err := NewCache(tempSQLiteFile)
	require.NoError(t, err)
	defer os.Remove(tempSQLiteFile)

	expected := []tensor.Prediction{{
		Label: "something cool",
		Score: 0.98,
	}, {
		Label: "something uncool",
		Score: 0.18,
	}}

	inception := func() ([]tensor.Prediction, error) {
		return expected, nil
	}

	actual, err := c.Inception("dummy-filename.jpg", "pnasnet", inception)
	require.NoError(t, err)

	assert.Equal(t, expected, actual)

	rows, err := c.Query(`
select filename, network, label, score
from predictions
where filename=$1 and network=$2`, "dummy-filename.jpg", "pnasnet")
	require.NoError(t, err)

	var filename, network, label string
	var score float32

	assert.True(t, rows.Next())
	err = rows.Scan(&filename, &network, &label, &score)
	require.NoError(t, err)
	assert.Equal(t, "dummy-filename.jpg", filename)
	assert.Equal(t, "pnasnet", network)
	assert.Equal(t, "something cool", label)
	assert.Equal(t, float32(0.98), score)

	assert.True(t, rows.Next())
	err = rows.Scan(&filename, &network, &label, &score)
	require.NoError(t, err)
	assert.Equal(t, "dummy-filename.jpg", filename)
	assert.Equal(t, "pnasnet", network)
	assert.Equal(t, "something uncool", label)
	assert.Equal(t, float32(0.18), score)

	assert.False(t, rows.Next())
}
