package classifiers

import (
	"image"
	"sort"

	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/pkg/errors"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

type Model struct {
	model  *tf.SavedModel
	labels Labels

	ModelName, TagName      string
	InputLayer, OutputLayer string
	ImageMode               string
	Labels                  string
	ImageHeight, ImageWidth uint
	IndexCorrection         int
}

func (m *Model) Load() (func() error, error) {
	if m.Labels != "" {
		l, err := labelsFromFile(m.Labels, m.IndexCorrection)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to read labels from file %q", m.Labels)
		}
		m.labels = l
	}

	model, err := tf.LoadSavedModel(m.ModelName, []string{m.TagName}, nil)
	if err != nil {
		return nil, errors.Wrapf(err,
			"failed to load saved model %q / tag %q", m.ModelName, m.TagName)
	}

	m.model = model

	return model.Session.Close, nil
}

type Predictions struct {
	scores []float32
	labels Labels
}

type Prediction struct {
	Label string  `json:"label"`
	Score float32 `json:"score"`
}

func (p *Predictions) Best(n int) []Prediction {
	var ret []Prediction

	for i, score := range p.scores {
		ret = append(ret, Prediction{
			Label: p.labels.Get(i),
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

	for i, score := range p.scores {
		if score < threshold {
			continue
		}

		ret = append(ret, Prediction{
			Label: p.labels.Get(i),
			Score: score,
		})
	}

	return ret
}

func (m *Model) Inception(img image.Image) (*Predictions, error) {
	img = imageutils.Scaled(img, m.ImageHeight, m.ImageWidth)

	tensor, err := imageToTensor(img, m.ImageMode, m.ImageHeight, m.ImageWidth)
	if err != nil {
		return nil, errors.Wrap(err, "error converting image to tensor")
	}

	result, err := m.model.Session.Run(
		map[tf.Output]*tf.Tensor{
			m.model.Graph.Operation(m.InputLayer).Output(0): tensor,
		},
		[]tf.Output{
			m.model.Graph.Operation(m.OutputLayer).Output(0),
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

	return &Predictions{
		scores: res[0],
		labels: m.labels}, nil
}

const (
	ImageModeTensorflow         = "tf"
	ImageModeTensorflowPositive = "tf-pos"
	ImageModeCaffe              = "caffe"
)

func imageToTensor(img image.Image, imageMode string, imageHeight, imageWidth uint) (*tf.Tensor, error) {
	switch imageMode {
	case ImageModeTensorflow:
		return imageToTensorTF(img, imageHeight, imageWidth)
	case ImageModeTensorflowPositive:
		return imageToTensorTFPositive(img, imageHeight, imageWidth)
	case ImageModeCaffe:
		return imageToTensorCaffe(img, imageHeight, imageWidth)
	}

	return nil, errors.Errorf("unknown image mode %q", imageMode)
}

func imageToTensorTF(img image.Image, imageHeight, imageWidth uint) (*tf.Tensor, error) {
	var image [1][][][3]float32

	for j := 0; j < int(imageHeight); j++ {
		image[0] = append(image[0], make([][3]float32, imageWidth))
	}

	for i := 0; i < int(imageWidth); i++ {
		for j := 0; j < int(imageHeight); j++ {
			r, g, b, _ := img.At(i, j).RGBA()
			image[0][j][i][0] = convertTF(r)
			image[0][j][i][1] = convertTF(g)
			image[0][j][i][2] = convertTF(b)
		}
	}

	return tf.NewTensor(image)
}

func convertTF(value uint32) float32 {
	return (float32(value>>8) - float32(127.5)) / float32(127.5)
}

func imageToTensorTFPositive(img image.Image, imageHeight, imageWidth uint) (*tf.Tensor, error) {
	var image [1][][][3]float32

	for j := 0; j < int(imageHeight); j++ {
		image[0] = append(image[0], make([][3]float32, imageWidth))
	}

	for i := 0; i < int(imageWidth); i++ {
		for j := 0; j < int(imageHeight); j++ {
			r, g, b, _ := img.At(i, j).RGBA()
			image[0][j][i][0] = convertTFPositive(r)
			image[0][j][i][1] = convertTFPositive(g)
			image[0][j][i][2] = convertTFPositive(b)
		}
	}

	return tf.NewTensor(image)
}

func convertTFPositive(value uint32) float32 {
	return float32(value>>8) / float32(255)
}

func imageToTensorCaffe(img image.Image, imageHeight, imageWidth uint) (*tf.Tensor, error) {
	var image [1][][][3]float32

	for j := 0; j < int(imageHeight); j++ {
		image[0] = append(image[0], make([][3]float32, imageWidth))
	}

	for i := 0; i < int(imageWidth); i++ {
		for j := 0; j < int(imageHeight); j++ {
			r, g, b, _ := img.At(i, j).RGBA()
			image[0][j][i][0] = convertCaffe(b) - 103.939
			image[0][j][i][1] = convertCaffe(g) - 116.779
			image[0][j][i][2] = convertCaffe(r) - 123.68
		}
	}

	return tf.NewTensor(image)
}

func convertCaffe(value uint32) float32 {
	return float32(value >> 8)
}
