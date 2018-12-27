package gildasai

type PredictionItem struct {
	Identifier  string
	Predictions []Prediction
}

type PredictionStore interface {
	Get(query, after string, n int) ([]PredictionItem, error)
}
