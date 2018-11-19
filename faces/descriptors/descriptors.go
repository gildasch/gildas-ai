package descriptors

import (
	"image"

	"github.com/nfnt/resize"
	"github.com/pkg/errors"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

type Descriptor struct {
	graph   *tf.Graph
	session *tf.Session
}

func NewDescriptor() (*Descriptor, error) {
	return NewDescriptorFromFile("descriptorsnet", "myTag")
}

func NewDescriptorFromFile(modelName, tagName string) (*Descriptor, error) {
	model, err := tf.LoadSavedModel(modelName, []string{tagName}, nil)
	if err != nil {
		return nil, errors.Wrapf(err,
			"failed to load saved model %q / tag %q", modelName, tagName)
	}

	return &Descriptor{
		graph:   model.Graph,
		session: model.Session,
	}, nil
}

func (d *Descriptor) Close() error {
	return d.session.Close()
}

func (d *Descriptor) Compute(img image.Image) (*Descriptors, error) {
	img = resize.Resize(150, 150, img, resize.NearestNeighbor)

	tensor, err := imageToTensor(img, uint(img.Bounds().Dy()), uint(img.Bounds().Dx()))
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
		return nil, errors.New("descriptors are empty")
	}

	out := &Descriptors{}

	for i, v := range res[0] {
		out[i] = v
	}

	return out, nil
}

func imageToTensor(img image.Image, imageHeight, imageWidth uint) (*tf.Tensor, error) {
	var image [1][][][3]float32

	for j := 0; j < int(imageHeight); j++ {
		image[0] = append(image[0], make([][3]float32, imageWidth))
	}

	for i := 0; i < int(imageWidth); i++ {
		for j := 0; j < int(imageHeight); j++ {
			r, g, b, _ := img.At(i, j).RGBA()
			image[0][j][i][0] = convert(r, 122.782)
			image[0][j][i][1] = convert(g, 117.001)
			image[0][j][i][2] = convert(b, 104.298)
		}
	}

	return tf.NewTensor(image)
}

func convert(value uint32, mean float32) float32 {
	return (float32(value>>8) - mean) / float32(255)
}

type Descriptors [128]float32

func (d *Descriptors) DistanceTo(d2 *Descriptors) float32 {
	sum := float32(0)

	for i := 0; i < 128; i++ {
		sum += (d[i] - d2[i]) * (d[i] - d2[i])
	}

	return sum
}
