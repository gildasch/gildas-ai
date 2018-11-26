package imageutils

import (
	"bytes"
	"io/ioutil"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFromZip(t *testing.T) {
	zipBytes, err := ioutil.ReadFile("test.zip")
	require.NoError(t, err)

	images, errs := FromZip(bytes.NewReader(zipBytes), int64(len(zipBytes)))
	assert.Len(t, errs, 0)
	assert.Len(t, images, 4)
}
