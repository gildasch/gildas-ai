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

type FaceItem struct {
	Identifier  string
	Network     string
	Detection   Detection
	Landmarks   Landmarks
	Descriptors Descriptors
}

type FaceStore interface {
	StoreFace(item *FaceItem) error
	GetFaces(id string) ([]*FaceItem, bool, error)
	GetAllFaces() ([]*FaceItem, error)
}
