package faces

import (
	"image"
	"image/draw"

	"github.com/gildasch/gildas-ai/faces/descriptors"
	"github.com/gildasch/gildas-ai/faces/descriptors/faceapi"
	"github.com/gildasch/gildas-ai/faces/detection"
	"github.com/gildasch/gildas-ai/faces/landmarks"
	"github.com/pkg/errors"
)

type Extractor struct {
	Detector   *detection.Detector
	Landmark   *landmarks.Landmark
	Descriptor *faceapi.Descriptor
}

func NewDefaultExtractor(modelRoot string) (*Extractor, error) {
	detector, err := detection.NewDetectorFromFile(modelRoot + "/detection/frozen_inference_graph_face.pb")
	if err != nil {
		return nil, err
	}

	landmark, err := landmarks.NewLandmarkFromFile(modelRoot+"/landmarks/landmarksnet", "myTag")
	if err != nil {
		return nil, err
	}

	descriptor, err := faceapi.NewDescriptorFromFile(modelRoot+"/descriptors/faceapi/descriptorsnet", "myTag")
	if err != nil {
		return nil, err
	}

	return &Extractor{
		Detector:   detector,
		Landmark:   landmark,
		Descriptor: descriptor,
	}, nil
}

func (e *Extractor) Extract(img image.Image) ([]image.Image, []descriptors.Descriptors, error) {
	allDetections, err := e.Detector.Detect(img)
	if err != nil {
		return nil, nil, errors.Wrap(err, "error detecting faces")
	}

	detections := allDetections.Above(0.6)

	images := []image.Image{}
	descrs := []descriptors.Descriptors{}
	for _, box := range detections.Boxes {
		if box.Dx() < 45 || box.Dy() < 45 {
			continue // face is too small
		}

		cropped := image.NewRGBA(box)
		draw.Draw(cropped, box, img, box.Min, draw.Src)

		landmarks, err := e.Landmark.Detect(cropped)
		if err != nil {
			return nil, nil, errors.Wrap(err, "error detecting landmarks")
		}

		cropped2 := landmarks.Center(cropped, img)

		descriptors, err := e.Descriptor.Compute(cropped2)
		if err != nil {
			return nil, nil, errors.Wrap(err, "error computing descriptors")
		}

		images = append(images, cropped2)
		descrs = append(descrs, descriptors)
	}

	return images, descrs, nil
}

func (e *Extractor) ExtractLandmarks(img image.Image) ([][]image.Point, []image.Image, error) {
	allDetections, err := e.Detector.Detect(img)
	if err != nil {
		return nil, nil, errors.Wrap(err, "error detecting faces")
	}

	detections := allDetections.Above(0.4)

	if detections.NumDetections == 0 {
		return nil, nil, errors.New("no face detected")
	}

	var ret [][]image.Point
	var crops []image.Image
	for _, box := range detections.Boxes {
		if box.Dx() < 45 || box.Dy() < 45 {
			continue // face is too small
		}

		cropped := image.NewRGBA(box)
		draw.Draw(cropped, box, img, box.Min, draw.Src)

		landmarks, err := e.Landmark.Detect(cropped)
		if err != nil {
			return nil, nil, errors.Wrap(err, "error detecting landmarks")
		}

		ret = append(ret, landmarks.PointsOnImage(cropped))
		crops = append(crops, cropped)
	}

	return ret, crops, nil
}
