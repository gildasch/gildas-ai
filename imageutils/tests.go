package imageutils

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"io/ioutil"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func AssertImageEqual(t *testing.T, expectedFilename string, actual image.Image) bool {
	start := time.Now()
	defer func() {
		fmt.Println("assertImageEqual in ", time.Since(start))
	}()

	expectedBytes, err := ioutil.ReadFile(expectedFilename)
	if err != nil {
		t.Errorf("could not read expected image %q: %v", expectedFilename, err)
		t.FailNow()
		return false
	}

	var actualBytes bytes.Buffer
	err = png.Encode(&actualBytes, actual)
	if err != nil {
		t.Errorf("could not encode actual image: %v", err)
		t.FailNow()
		return false
	}

	return assert.Equal(t, expectedBytes, actualBytes.Bytes())
}
