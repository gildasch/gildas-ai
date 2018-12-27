package listing

import gildasai "github.com/gildasch/gildas-ai"

type Listing struct {
	Store Store
}

type Item struct {
	Filename    string
	Predictions []gildasai.Prediction
}

type Store interface {
	Get(query, after string, n int) ([]Item, error)
}
