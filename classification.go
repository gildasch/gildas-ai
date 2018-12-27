package gildasai

import (
	"image"
	"sort"
)

type Predictions []Prediction

type Prediction struct {
	Score float32
	Label string
}

func (p Predictions) Best(n int) []Prediction {
	sort.Slice(p, func(i, j int) bool {
		return p[i].Score > p[j].Score
	})

	return p[:n]
}

func (p Predictions) Above(threshold float32) []Prediction {
	var ret []Prediction

	for _, pp := range p {
		if pp.Score < threshold {
			continue
		}

		ret = append(ret, pp)
	}

	return ret
}

type Classifier interface {
	Classify(input image.Image) (Predictions, error)
}
