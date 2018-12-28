package gildasai

type PredictionItem struct {
	Identifier  string
	Predictions Predictions
}

type PredictionStore interface {
	GetPrediction(id string) (*PredictionItem, bool, error)
	StorePrediction(id string, item *PredictionItem) error
	SearchPrediction(query, after string, n int) ([]*PredictionItem, error)
}
