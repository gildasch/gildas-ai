package tensor

import (
	"image"

	"github.com/pkg/errors"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

type Model struct {
	model *tf.SavedModel
}

func NewModel(modelName, tagName string) (*Model, func() error, error) {
	model, err := tf.LoadSavedModel(modelName, []string{tagName}, nil)
	if err != nil {
		return nil, nil, errors.Wrapf(err,
			"failed to load saved model %q / tag %q", modelName, tagName)
	}

	return &Model{model: model}, model.Session.Close, nil
}

type Predictions []float32

func (p Predictions) Best() (i int, v float32) {
	best, bestV := 0, float32(0.0)
	for i, v := range p {
		if bestV < v {
			best = i
			bestV = v
		}
	}

	return best, bestV
}

func (m *Model) Inception(img image.Image) (Predictions, error) {
	tensor, err := imageToTensor(img)
	if err != nil {
		return nil, errors.Wrap(err, "error converting image to tensor")
	}

	result, err := m.model.Session.Run(
		map[tf.Output]*tf.Tensor{
			m.model.Graph.Operation("input_1").Output(0): tensor,
		},
		[]tf.Output{
			m.model.Graph.Operation("predictions/Softmax").Output(0),
		},
		nil,
	)
	if err != nil {
		return nil, errors.Wrap(err, "error running the model session")
	}

	if len(result) < 1 {
		return nil, errors.New("result is empty")
	}

	res, ok := result[0].Value().([][]float32)
	if !ok {
		return nil, errors.Errorf("result has unexpected type %T", result[0].Value())
	}

	if len(res) < 1 {
		return nil, errors.New("predictions are empty")
	}

	return res[0], nil
}

func imageToTensor(img image.Image) (*tf.Tensor, error) {
	var image [1][299][299][3]float32
	for i := 0; i < 299; i++ {
		for j := 0; j < 299; j++ {
			r, g, b, _ := img.At(i, j).RGBA()
			image[0][i][j][0] = convert(r)
			image[0][i][j][1] = convert(g)
			image[0][i][j][2] = convert(b)
		}
	}

	return tf.NewTensor(image)
}

func convert(value uint32) float32 {
	return (float32(value>>8) - float32(127.5)) / float32(127.5)
}
