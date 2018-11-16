package landmarks

import (
	"fmt"
	"testing"

	"github.com/gildasch/gildas-ai/image"
	"github.com/stretchr/testify/require"
)

func TestDetection(t *testing.T) {
	l, err := NewLandmark()
	require.NoError(t, err)

	testImage, err := image.FromFile("1.jpg")
	require.NoError(t, err)

	landmarks, err := l.Detect(testImage)
	require.NoError(t, err)

	fmt.Println(landmarks)
}
