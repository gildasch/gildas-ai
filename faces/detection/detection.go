package detection

import (
	"image"
	"io/ioutil"

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

type Detections struct {
	Boxes         []image.Rectangle
	Scores        []float32
	Classes       []float32
	NumDetections int
}

func (d Detections) Above(threshold float32) Detections {
	filtered := Detections{}

	for i := 0; i < d.NumDetections; i++ {
		if d.Scores[i] < threshold {
			break
		}

		filtered.Boxes = append(filtered.Boxes, d.Boxes[i])
		filtered.Scores = append(filtered.Scores, d.Scores[i])
		filtered.Classes = append(filtered.Classes, d.Classes[i])
		filtered.NumDetections++
	}

	return filtered
}

func (d *Detector) Detect(img image.Image) (*Detections, error) {
	tensor, err := imageToTensor(img, uint(img.Bounds().Dy()), uint(img.Bounds().Dx()))
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

	detections := &Detections{}

	boxBatches, ok := result[0].Value().([][][]float32)
	if !ok || len(boxBatches) < 1 {
		return nil, errors.New("detection_boxes has unexprected shape")
	}
	boxes := boxBatches[0]

	for _, box := range boxes {
		detections.Boxes = append(detections.Boxes, image.Rectangle{
			Min: image.Point{
				X: int(float32(img.Bounds().Max.X) * box[1]),
				Y: int(float32(img.Bounds().Max.Y) * box[0]),
			},
			Max: image.Point{
				X: int(float32(img.Bounds().Max.X) * box[3]),
				Y: int(float32(img.Bounds().Max.Y) * box[2]),
			},
		})
	}

	scoreBatches, ok := result[1].Value().([][]float32)
	if !ok || len(scoreBatches) < 1 {
		return nil, errors.Errorf("detection_scores has unexprected shape %T", result[0].Value())
	}
	scores := scoreBatches[0]

	for _, score := range scores {
		detections.Scores = append(detections.Scores, score)
	}

	classBatches, ok := result[2].Value().([][]float32)
	if !ok || len(classBatches) < 1 {
		return nil, errors.New("detection_classes has unexprected shape")
	}
	classes := classBatches[0]

	for _, class := range classes {
		detections.Classes = append(detections.Classes, class)
	}

	numDetectionBatches, ok := result[3].Value().([]float32)
	if !ok || len(classBatches) < 1 {
		return nil, errors.New("num_detections has unexprected shape")
	}
	detections.NumDetections = int(numDetectionBatches[0])

	return detections, nil
}

func imageToTensor(img image.Image, imageHeight, imageWidth uint) (*tf.Tensor, error) {
	var image [1][][][3]uint8

	for j := 0; j < int(imageHeight); j++ {
		image[0] = append(image[0], make([][3]uint8, imageWidth))
	}

	for i := 0; i < int(imageWidth); i++ {
		for j := 0; j < int(imageHeight); j++ {
			r, g, b, _ := img.At(i, j).RGBA()
			image[0][j][i][0] = convert(r)
			image[0][j][i][1] = convert(g)
			image[0][j][i][2] = convert(b)
		}
	}

	return tf.NewTensor(image)
}

func convert(value uint32) uint8 {
	return uint8(value >> 8)
}
