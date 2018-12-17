package mask

import (
	"image"
	"image/color"
	"image/draw"
	"math"

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

	return &RCNN{model: model}, nil
}

func (r *RCNN) Close() error {
	return r.model.Session.Close()
}

type Detections struct {
	Values [][][]float32
}

type Masks struct {
	Values [][][][][]float32
}

var classes = []string{
	"BG", "person", "bicycle", "car", "motorcycle", "airplane",
	"bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird",
	"cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
	"zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
	"suitcase", "frisbee", "skis", "snowboard", "sports ball",
	"kite", "baseball bat", "baseball glove", "skateboard",
	"surfboard", "tennis racket", "bottle", "wine glass", "cup",
	"fork", "knife", "spoon", "bowl", "banana", "apple",
	"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
	"donut", "cake", "chair", "couch", "potted plant", "bed",
	"dining table", "toilet", "tv", "laptop", "mouse", "remote",
	"keyboard", "cell phone", "microwave", "oven", "toaster",
	"sink", "refrigerator", "book", "clock", "vase", "scissors",
	"teddy bear", "hair drier", "toothbrush"}

var colors = []color.Color{
	color.NRGBA{R: 255, A: 128},
	color.NRGBA{G: 255, A: 128},
	color.NRGBA{B: 255, A: 128},
	color.NRGBA{R: 255, G: 255, A: 128},
	color.NRGBA{G: 255, B: 255, A: 128},
	color.NRGBA{B: 255, R: 255, A: 128},
}

func (m *Masks) DrawAllOnImage(detections *Detections, img image.Image) image.Image {
	for i := range detections.Values[0] {
		box := image.Rectangle{
			Min: image.Point{
				X: int(detections.Values[0][i][1] * float32(img.Bounds().Dx())),
				Y: int(detections.Values[0][i][0] * float32(img.Bounds().Dy())),
			},
			Max: image.Point{
				X: int(detections.Values[0][i][3] * float32(img.Bounds().Dx())),
				Y: int(detections.Values[0][i][2] * float32(img.Bounds().Dy())),
			},
		}
		classID := int(detections.Values[0][i][4])
		score := detections.Values[0][i][5]

		if score == 0 {
			break
		}

		img = m.DrawOnImage(i, classID, box, img, colors[i])
	}

	return img
}

func (m *Masks) DrawOnImage(detectionID, classID int, box image.Rectangle, img image.Image, c color.Color) image.Image {
	withMask := image.NewNRGBA(img.Bounds())
	draw.Draw(withMask, img.Bounds(), img, image.ZP, draw.Src)

	mask := image.NewNRGBA(image.Rect(0, 0, 28, 28))
	for x := mask.Bounds().Min.X; x < mask.Bounds().Max.X; x++ {
		for y := mask.Bounds().Min.Y; y < mask.Bounds().Max.Y; y++ {
			if m.Values[0][detectionID][y][x][classID] > 0.5 {
				mask.Set(x, y, c)
			}
		}
	}
	maskResized := imageutils.Scaled(mask, uint(box.Bounds().Dx()), uint(box.Bounds().Dy()))

	draw.Draw(withMask, box.Bounds(), maskResized, image.ZP, draw.Over)

	for x := box.Bounds().Min.X; x < box.Bounds().Max.X; x++ {
		withMask.Set(x, box.Bounds().Min.Y, color.NRGBA{R: 255})
		withMask.Set(x, box.Bounds().Max.Y, color.NRGBA{R: 255})
	}

	for y := box.Bounds().Min.Y; y < box.Bounds().Max.Y; y++ {
		withMask.Set(box.Bounds().Min.X, y, color.NRGBA{R: 255})
		withMask.Set(box.Bounds().Max.X, y, color.NRGBA{R: 255})
	}

	return withMask
}

func (r *RCNN) Inception(img image.Image) (*Detections, *Masks, error) {
	imgTensor, meta, anchors, err := makeInputs(img)
	if err != nil {
		return nil, nil, errors.Wrap(err, "error converting image to tensor")
	}

	result, err := r.model.Session.Run(
		map[tf.Output]*tf.Tensor{
			r.model.Graph.Operation("input_image").Output(0):      imgTensor,
			r.model.Graph.Operation("input_image_meta").Output(0): meta,
			r.model.Graph.Operation("input_anchors").Output(0):    anchors,
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
		return nil, nil, errors.Wrap(err, "error running the model session")
	}

	if len(result) < 1 {
		return nil, nil, errors.New("result is empty")
	}

	detections, ok := result[0].Value().([][][]float32)
	if !ok {
		return nil, nil, errors.Errorf("result has unexpected type %T", result[0].Value())
	}

	if len(detections) < 1 {
		return nil, nil, errors.New("predictions are empty")
	}

	masks, ok := result[3].Value().([][][][][]float32)
	if !ok {
		return nil, nil, errors.Errorf("result has unexpected type %T", result[3].Value())
	}

	if len(masks) < 1 {
		return nil, nil, errors.New("predictions are empty")
	}

	return &Detections{Values: detections}, &Masks{Values: masks}, nil
}

func makeInputs(img image.Image) (imgTensor, meta, anchors *tf.Tensor, err error) {
	resized := imageutils.Scaled(img, 1024, 1024)

	imgTensor, err = imageToTensor(resized)
	if err != nil {
		return nil, nil, nil, err
	}

	meta, err = composeImageMeta(0, img.Bounds(), resized.Bounds(), resized.Bounds(),
		float32(resized.Bounds().Dy())/float32(img.Bounds().Dy()), 81)
	if err != nil {
		return nil, nil, nil, err
	}

	anchors, err = getAnchors(resized.Bounds())
	if err != nil {
		return nil, nil, nil, err
	}

	return imgTensor, meta, anchors, nil
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
			image[0][j][i][0] = convert(r) - 123.7
			image[0][j][i][1] = convert(g) - 116.8
			image[0][j][i][2] = convert(b) - 103.9
		}
	}

	return tf.NewTensor(image)
}

func convert(value uint32) float32 {
	return float32(value >> 8)
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

func getAnchors(imageBounds image.Rectangle) (*tf.Tensor, error) {
	backboneStrides := []int{4, 8, 16, 32, 64}
	backboneShapes := computeBackboneShapes(imageBounds, backboneStrides)

	anchorScales := []int{32, 64, 128, 256, 512}
	anchorRatios := []float32{0.5, 1, 2}
	anchorStride := 1

	a := generatePyramidAnchors(
		anchorScales,
		anchorRatios,
		backboneShapes,
		backboneStrides,
		anchorStride)

	normalizeAnchors(a, imageBounds)

	return tf.NewTensor(a)
}

func computeBackboneShapes(imageBounds image.Rectangle, backboneStrides []int) [][]float32 {
	var backboneShapes [][]float32
	for _, s := range backboneStrides {
		backboneShapes = append(backboneShapes, []float32{
			float32(math.Ceil(float64(imageBounds.Dy()) / float64(s))),
			float32(math.Ceil(float64(imageBounds.Dx()) / float64(s))),
		})
	}

	return backboneShapes
}

func generatePyramidAnchors(
	scales []int,
	ratios []float32,
	featureShapes [][]float32,
	featureStrides []int,
	anchorStride int) [1][][4]float32 {
	/*
	   Generate anchors at different levels of a feature pyramid. Each scale
	   is associated with a level of the pyramid, but each ratio is used in
	   all levels of the pyramid.

	   Returns:
	   anchors: [N, (y1, x1, y2, x2)]. All generated anchors in one array. Sorted
	       with the same order of the given scales. So, anchors of scale[0] come
	       first, then anchors of scale[1], and so on.
	*/

	var anchors [1][][4]float32
	for i := range scales {
		anchors[0] = append(anchors[0], generateAnchors(
			scales[i],
			ratios,
			featureShapes[i],
			featureStrides[i],
			anchorStride)...)
	}

	return anchors
}

func generateAnchors(
	scales int,
	ratios []float32,
	shape []float32,
	featureStride int,
	anchorStride int) [][4]float32 {
	/*
	   scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
	   ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
	   shape: [height, width] spatial shape of the feature map over which
	           to generate anchors.
	   feature_stride: Stride of the feature map relative to the image in pixels.
	   anchor_stride: Stride of anchors on the feature map. For example, if the
	       value is 2 then generate anchors for every other feature map pixel.

	*/

	heights := []float32{
		float32(scales) / float32(math.Sqrt(float64(ratios[0]))),
		float32(scales) / float32(math.Sqrt(float64(ratios[1]))),
		float32(scales) / float32(math.Sqrt(float64(ratios[2]))),
	}
	widths := []float32{
		float32(scales) * float32(math.Sqrt(float64(ratios[0]))),
		float32(scales) * float32(math.Sqrt(float64(ratios[1]))),
		float32(scales) * float32(math.Sqrt(float64(ratios[2]))),
	}

	var boxes [][4]float32
	for j := float32(0); j < shape[0]; j += float32(anchorStride) {
		for i := float32(0); i < shape[1]; i += float32(anchorStride) {
			for k := 0; k < 3; k++ {
				boxes = append(boxes, [4]float32{
					j*float32(featureStride) - 0.5*heights[k],
					i*float32(featureStride) - 0.5*widths[k],
					j*float32(featureStride) + 0.5*heights[k],
					i*float32(featureStride) + 0.5*widths[k],
				})
			}
		}
	}

	return boxes
}

/*
Example execution of generateAnchors (from Python)

ratios : [0.5, 1, 2]
scales : 32
feature_stride : 4
anchor_stride : 1

ratios : [0.5, 1, 2]
scales : [32, 32, 32]

shape : [256, 256]
heights : [45.2548, 32, 22.6274]
widths : [22.6274, 32, 45.2548]

shifts_x : [0, 4, 8, 12, .., 1016, 1020] (size 256)
shifts_y : [0, 4, 8, 12, .., 1016, 1020] (size 256)

shifts_x : [256*[0, 4, 8, 12, .., 1016, 1020] (size 256)] total: 65536
shifts_y : [25*[0, 4, 8, 12, .., 1016, 1020] (size 256)]

box_widths : [65536*[22.6274, 32, 45.2548]]
box_centers_x : [[0,0,0],[4,4,4],[8,8,8]...[1020,1020,1020] * 256]

box_heights : [65536*[45.2548, 32, 22.6274]]
box_centers_y : [[0,0,0]*256, [4,4,4]*256,...,[1020,1020,1020]*256]

box_centers : [[0,0]*3,[0,4]*3,...[0,1020]*3,[4,0]*3,[4,4]*3,...] size 196608
box_sizes : [[45.2548, 22.6274], [32, 32], [22.6274, 45.2548] * 65536] size 196608

boxes : [
         [-22.6274, -11.3137, 22.6274, 11.3137],
         [-16, -16, 16, 16],
         [-11.3137, -22.6274 , 11.3137, 22.6274],
         [-22.6274, -7.3137, 22.6274, 15.3137],
         [-16, -12, 16, 20],
         [-11.3137, -18.6274 , 11.3137, 26.6274],
         [-22.6274, -3.3137, 22.6274, 19.3137],
         [-16, -8, 16, 24],
         [-11.3137, -14.6274 , 11.3137, 30.6274],
         ...
         ] size 196608
boxes[12345] : [41.3725, 64.6862, 86.6274, 87.3137]
boxes[123456] : [617.3725, 756.6862, 662.6274, 779.3137]

boxes[196605] : 997.3725, 1008.6862, 1042.6274, 1031.3137
boxes[196606] : 1004, 1004, 1036, 1036
boxes[196607] : 1008.6862, 997.3725, 1031.3137, 1042.6274
*/

func normalizeAnchors(a [1][][4]float32, imageBounds image.Rectangle) {
	for i := range a[0] {
		a[0][i][0] = a[0][i][0] / float32(imageBounds.Dy())
		a[0][i][1] = a[0][i][1] / float32(imageBounds.Dx())
		a[0][i][2] = (a[0][i][2] - 1) / float32(imageBounds.Dy())
		a[0][i][3] = (a[0][i][3] - 1) / float32(imageBounds.Dx())
	}
}
