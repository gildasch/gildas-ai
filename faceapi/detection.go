package faceapi

import (
	"image"
	"io/ioutil"

	gildasai "github.com/gildasch/gildas-ai"
	"github.com/pkg/errors"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

type Detector struct {
	graph   *tf.Graph
	session *tf.Session
}

func NewDetector() (*Detector, error) {
	return NewDetectorFromFile("frozen_inference_graph_face.pb")
}

func NewDetectorFromFile(modelFilename string) (*Detector, error) {
	model, err := ioutil.ReadFile(modelFilename)
	if err != nil {
		return nil, errors.Wrapf(err, "error reading model file %q", modelFilename)
	}

	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		return nil, errors.Wrapf(err, "error importing model to new graph")
	}

	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, errors.Wrapf(err, "error creating new session from graph")
	}

	return &Detector{
		graph:   graph,
		session: session,
	}, nil
}

func (d *Detector) Close() error {
	return d.session.Close()
}

func (d *Detector) Detect(img image.Image) ([]gildasai.Detection, error) {
	tensor, err := imageToTensorDetection(img, uint(img.Bounds().Dy()), uint(img.Bounds().Dx()))
	if err != nil {
		return nil, errors.Wrap(err, "error converting image to tensor")
	}

	result, err := d.session.Run(
		map[tf.Output]*tf.Tensor{
			d.graph.Operation("image_tensor").Output(0): tensor,
		},
		[]tf.Output{
			d.graph.Operation("detection_boxes").Output(0),
			d.graph.Operation("detection_scores").Output(0),
			d.graph.Operation("detection_classes").Output(0),
			d.graph.Operation("num_detections").Output(0),
		},
		nil)
	if err != nil {
		return nil, errors.Wrap(err, "error running the tensorflow session")
	}

	boxBatches, ok := result[0].Value().([][][]float32)
	if !ok || len(boxBatches) < 1 {
		return nil, errors.New("detection_boxes has unexprected shape")
	}
	boxes := boxBatches[0]

	scoreBatches, ok := result[1].Value().([][]float32)
	if !ok || len(scoreBatches) < 1 {
		return nil, errors.Errorf("detection_scores has unexprected shape %T", result[0].Value())
	}
	scores := scoreBatches[0]

	classBatches, ok := result[2].Value().([][]float32)
	if !ok || len(classBatches) < 1 {
		return nil, errors.New("detection_classes has unexprected shape")
	}
	classes := classBatches[0]

	numDetectionsBatches, ok := result[3].Value().([]float32)
	if !ok || len(classBatches) < 1 {
		return nil, errors.New("num_detections has unexprected shape")
	}
	numDetections := int(numDetectionsBatches[0])

	var detections []gildasai.Detection
	for i := 0; i < numDetections; i++ {
		detections = append(detections, gildasai.Detection{
			Box: image.Rectangle{
				Min: image.Point{
					X: int(float32(img.Bounds().Max.X) * boxes[i][1]),
					Y: int(float32(img.Bounds().Max.Y) * boxes[i][0]),
				},
				Max: image.Point{
					X: int(float32(img.Bounds().Max.X) * boxes[i][3]),
					Y: int(float32(img.Bounds().Max.Y) * boxes[i][2]),
				},
			},
			Score: scores[i],
			Class: classes[i],
		})
	}

	return detections, nil
}

func imageToTensorDetection(img image.Image, imageHeight, imageWidth uint) (*tf.Tensor, error) {
	var image [1][][][3]uint8

	for j := 0; j < int(imageHeight); j++ {
		image[0] = append(image[0], make([][3]uint8, imageWidth))
	}

	for i := 0; i < int(imageWidth); i++ {
		for j := 0; j < int(imageHeight); j++ {
			r, g, b, _ := img.At(i, j).RGBA()
			image[0][j][i][0] = convertDetection(r)
			image[0][j][i][1] = convertDetection(g)
			image[0][j][i][2] = convertDetection(b)
		}
	}

	return tf.NewTensor(image)
}

func convertDetection(value uint32) uint8 {
	return uint8(value >> 8)
}
