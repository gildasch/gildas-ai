package objects

import (
	"sort"

	"github.com/gildasch/gildas-ai/objects/labels"
)

type Predictions struct {
	Scores []float32
	Labels labels.Labels
}

type Prediction struct {
	Label string  `json:"label"`
	Score float32 `json:"score"`
}

func (p *Predictions) Best(n int) []Prediction {
	var ret []Prediction

	for i, score := range p.Scores {
		ret = append(ret, Prediction{
			Label: p.Labels.Get(i),
			Score: score,
		})
	}

	sort.Slice(ret, func(i, j int) bool {
		return ret[i].Score > ret[j].Score
	})

	return ret[:n]
}

func (p *Predictions) Above(threshold float32) []Prediction {
	var ret []Prediction

	for i, score := range p.Scores {
		if score < threshold {
			continue
		}

		ret = append(ret, Prediction{
			Label: p.Labels.Get(i),
			Score: score,
		})
	}

	return ret
}
