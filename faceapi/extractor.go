package faceapi

import gildasai "github.com/gildasch/gildas-ai"

func NewDefaultExtractor(modelRoot string) (*gildasai.Extractor, error) {
	detector, err := NewDetectorFromFile(modelRoot + "/frozen_inference_graph_face.pb")
	if err != nil {
		return nil, err
	}

	landmark, err := NewLandmarkFromFile(modelRoot+"/landmarksnet", "myTag")
	if err != nil {
		return nil, err
	}

	descriptor, err := NewDescriptorFromFile(modelRoot+"/descriptorsnet", "myTag")
	if err != nil {
		return nil, err
	}

	return &gildasai.Extractor{
		Detector:   detector,
		Landmark:   landmark,
		Descriptor: descriptor,
	}, nil
}
