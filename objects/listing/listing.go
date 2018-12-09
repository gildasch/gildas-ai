package listing

import "github.com/gildasch/gildas-ai/objects"

type Listing struct {
	Store Store
}

type Item struct {
	Filename    string
	Predictions []objects.Prediction
}

type Store interface {
	Get(query, after string, n int) ([]Item, error)
}
