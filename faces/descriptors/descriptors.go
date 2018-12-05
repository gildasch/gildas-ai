package descriptors

import (
	"math"

	"github.com/pkg/errors"
)

type Descriptors []float32

func (d Descriptors) DistanceTo(d2 Descriptors) (float32, error) {
	if len(d) != len(d2) {
		return 0, errors.Errorf(
			"cannot calculate distance between descriptors of dimensions %d and %d", len(d), len(d2))
	}

	sum := float32(0)

	for i := 0; i < len(d); i++ {
		sum += (d[i] - d2[i]) * (d[i] - d2[i])
	}

	return float32(math.Sqrt(float64(sum))), nil
}
