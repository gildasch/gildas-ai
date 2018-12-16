package mask

import (
	"image"

	"github.com/gildasch/gildas-ai/imageutils"
	"github.com/pkg/errors"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

type RCNN struct {
	model *tf.SavedModel
}

func NewRCNN(modelPath, tagName string) (*RCNN, error) {
	model, err := tf.LoadSavedModel(modelPath, []string{tagName}, nil)
	if err != nil {
		return nil, errors.Wrapf(err,
			"failed to load saved model %q / tag %q", modelPath, tagName)
	}

	// for _, op := range model.Graph.Operations() {
	// 	fmt.Println(op.Name())
	// }

	return &RCNN{model: model}, nil
}

func (r *RCNN) Close() error {
	return r.model.Session.Close()
}

func (r *RCNN) Inception(img image.Image) (*string, error) {
	imgTensor, meta, window, err := makeInputs(img)
	if err != nil {
		return nil, errors.Wrap(err, "error converting image to tensor")
	}

	result, err := r.model.Session.Run(
		map[tf.Output]*tf.Tensor{
			r.model.Graph.Operation("input_image").Output(0):      imgTensor,
			r.model.Graph.Operation("input_image_meta").Output(0): meta,
			r.model.Graph.Operation("input_anchors").Output(0):    window,
		},
		[]tf.Output{
			r.model.Graph.Operation("mrcnn_detection/Reshape_1").Output(0),
			r.model.Graph.Operation("mrcnn_class/Reshape_1").Output(0),
			r.model.Graph.Operation("mrcnn_bbox/Reshape").Output(0),
			r.model.Graph.Operation("mrcnn_mask/Reshape_1").Output(0),
			r.model.Graph.Operation("ROI/packed_2").Output(0),
			r.model.Graph.Operation("rpn_class/concat").Output(0),
			r.model.Graph.Operation("rpn_bbox/concat").Output(0),
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

	str := ""
	return &str, nil
}

func makeInputs(img image.Image) (imgTensor, meta, window *tf.Tensor, err error) {
	resized := imageutils.Scaled(img, 800, 800)

	imgTensor, err = imageToTensor(resized)
	if err != nil {
		return nil, nil, nil, err
	}

	meta, err = composeImageMeta(0, img.Bounds(), resized.Bounds(), resized.Bounds(),
		float32(resized.Bounds().Dy())/float32(img.Bounds().Dy()), 81)
	if err != nil {
		return nil, nil, nil, err
	}

	window, err = tf.NewTensor([1][]float32{[]float32{
		float32(resized.Bounds().Min.Y), float32(resized.Bounds().Min.X),
		float32(resized.Bounds().Max.Y), float32(resized.Bounds().Max.X),
	}})
	if err != nil {
		return nil, nil, nil, err
	}

	return imgTensor, meta, window, nil
}

func imageToTensor(img image.Image) (*tf.Tensor, error) {
	imageHeight, imageWidth := img.Bounds().Dy(), img.Bounds().Dx()

	var image [1][][][3]float32

	for j := 0; j < int(imageHeight); j++ {
		image[0] = append(image[0], make([][3]float32, imageWidth))
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

func convert(value uint32) float32 {
	return (float32(value>>8) - float32(127.5)) / float32(127.5)
}

func composeImageMeta(imageID int, originalBounds, resizedBounds, window image.Rectangle,
	scale float32, numClasses int) (*tf.Tensor, error) {
	var meta [1][]float32

	meta[0] = append(meta[0], float32(imageID))
	meta[0] = append(meta[0], float32(originalBounds.Dy()), float32(originalBounds.Dx()), 3)
	meta[0] = append(meta[0], float32(resizedBounds.Dy()), float32(resizedBounds.Dx()), 3)
	meta[0] = append(meta[0],
		float32(window.Min.Y), float32(window.Min.X),
		float32(window.Max.Y), float32(window.Max.X))
	meta[0] = append(meta[0], scale)
	meta[0] = append(meta[0], make([]float32, numClasses)...)

	return tf.NewTensor(meta)
}
