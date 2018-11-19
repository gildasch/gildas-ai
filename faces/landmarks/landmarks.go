package landmarks

import (
	"image"
	"image/color"
	"image/draw"

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

func (d *Landmark) Detect(img image.Image) (*Landmarks, error) {
	img = resize.Resize(112, 112, img, resize.NearestNeighbor)

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
		return nil, errors.New("landmarks are empty")
	}

	return &Landmarks{
		coords: res[0],
	}, nil
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

type Landmarks struct {
	coords []float32
}

func (l *Landmarks) PointsOnImage(img image.Image) []image.Point {
	w, h := float32(img.Bounds().Dx()), float32(img.Bounds().Dy())
	minX, minY := img.Bounds().Min.X, img.Bounds().Min.Y

	points := []image.Point{}
	for i := 0; i < len(l.coords)-1; i += 2 {
		points = append(points, image.Point{
			X: minX + int(w*l.coords[i]),
			Y: minY + int(h*l.coords[i+1]),
		})
	}

	return points
}

func (l *Landmarks) DrawOnImage(img image.Image) image.Image {
	out := image.NewRGBA(img.Bounds())

	draw.Draw(out, img.Bounds(), img, image.ZP, draw.Src)

	for _, p := range l.PointsOnImage(img) {
		drawPoint(out, p)
	}

	return out
}

func drawPoint(img *image.RGBA, p image.Point) {
	width := 3

	for i := p.X - width/2; i < p.X+width/2; i++ {
		for j := p.Y - width/2; j < p.Y+width/2; j++ {
			img.Set(i, j, color.RGBA{G: 255})
		}
	}
}

func (l *Landmarks) Center(img image.Image) image.Image {
	bounds := img.Bounds()
	minX, minY, maxX, maxY := bounds.Max.X, bounds.Max.Y, bounds.Min.X, bounds.Min.Y

	for _, p := range l.PointsOnImage(img) {
		if p.X < minX {
			minX = p.X
		}
		if p.Y < minY {
			minY = p.Y
		}
		if p.X > maxX {
			maxX = p.X
		}
		if p.Y > maxY {
			maxY = p.Y
		}
	}

	rect := image.Rectangle{
		Min: image.Point{
			X: minX,
			Y: minY,
		},
		Max: image.Point{
			X: maxX,
			Y: maxY,
		},
	}

	rect = square(rect)

	out := image.NewRGBA(rect)

	draw.Draw(out, out.Bounds(), img, rect.Min, draw.Src)

	return out
}

func square(rect image.Rectangle) image.Rectangle {
	width, height := rect.Max.X-rect.Min.X, rect.Max.Y-rect.Min.Y

	if height > width {
		left := (height - width) / 2
		right := height - width - left

		return image.Rectangle{
			Min: image.Point{
				X: rect.Min.X - left,
				Y: rect.Min.Y,
			},
			Max: image.Point{
				X: rect.Max.X + right,
				Y: rect.Max.Y,
			},
		}
	}

	if width > height {
		top := (width - height) / 2
		bottom := width - height - top

		return image.Rectangle{
			Min: image.Point{
				X: rect.Min.X,
				Y: rect.Min.Y - top,
			},
			Max: image.Point{
				X: rect.Max.X,
				Y: rect.Max.Y + bottom,
			},
		}
	}

	return rect
}
