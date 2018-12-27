package faceapi

import (
	"image"

	gildasai "github.com/gildasch/gildas-ai"
	"github.com/nfnt/resize"
	"github.com/pkg/errors"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

type Landmark struct {
	graph   *tf.Graph
	session *tf.Session
}

func NewLandmark() (*Landmark, error) {
	return NewLandmarkFromFile("landmarksnet", "myTag")
}

func NewLandmarkFromFile(modelName, tagName string) (*Landmark, error) {
	model, err := tf.LoadSavedModel(modelName, []string{tagName}, nil)
	if err != nil {
		return nil, errors.Wrapf(err,
			"failed to load saved model %q / tag %q", modelName, tagName)
	}

	return &Landmark{
		graph:   model.Graph,
		session: model.Session,
	}, nil
}

func (d *Landmark) Close() error {
	return d.session.Close()
}

func (d *Landmark) Detect(img image.Image) (*gildasai.Landmarks, error) {
	img = resize.Resize(112, 112, img, resize.NearestNeighbor)

	tensor, err := imageToTensorLandmarks(img, uint(img.Bounds().Dy()), uint(img.Bounds().Dx()))
	if err != nil {
		return nil, errors.Wrap(err, "error converting image to tensor")
	}

	result, err := d.session.Run(
		map[tf.Output]*tf.Tensor{
			d.graph.Operation("input").Output(0): tensor,
		},
		[]tf.Output{
			d.graph.Operation("output").Output(0),
		},
		nil)
	if err != nil {
		return nil, errors.Wrap(err, "error running the tensorflow session")
	}

	if len(result) < 1 {
		return nil, errors.New("result is empty")
	}

	res, ok := result[0].Value().([][]float32)
	if !ok {
		return nil, errors.Errorf("result has unexpected type %T", result[0].Value())
	}

	if len(res) < 1 {
		return nil, errors.New("landmarks are empty")
	}

	return &gildasai.Landmarks{
		Coords: res[0],
	}, nil
}

func imageToTensorLandmarks(img image.Image, imageHeight, imageWidth uint) (*tf.Tensor, error) {
	var image [1][][][3]float32

	for j := 0; j < int(imageHeight); j++ {
		image[0] = append(image[0], make([][3]float32, imageWidth))
	}

	for i := 0; i < int(imageWidth); i++ {
		for j := 0; j < int(imageHeight); j++ {
			r, g, b, _ := img.At(i, j).RGBA()
			image[0][j][i][0] = convertLandmarks(r, 122.782)
			image[0][j][i][1] = convertLandmarks(g, 117.001)
			image[0][j][i][2] = convertLandmarks(b, 104.298)
		}
	}

	return tf.NewTensor(image)
}

func convertLandmarks(value uint32, mean float32) float32 {
	return (float32(value>>8) - mean) / float32(255)
}
