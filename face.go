package gildasai

import (
	"image"
	"image/color"
	"image/draw"
	"math"

	"github.com/disintegration/imaging"
	"github.com/pkg/errors"
)

var (
	ErrNoFaceDetected = errors.New("no face detected")
)

type Detection struct {
	Box   image.Rectangle
	Score float32
	Class float32
}

func Above(allDetections []Detection, threshold float32) []Detection {
	var above []Detection

	for _, d := range allDetections {
		if d.Score < threshold {
			continue
		}
		above = append(above, d)
	}

	return above
}

type Landmarks struct {
	Coords []float32
}

func (l *Landmarks) PointsOnImage(img image.Image) []image.Point {
	w, h := float32(img.Bounds().Dx()), float32(img.Bounds().Dy())
	minX, minY := img.Bounds().Min.X, img.Bounds().Min.Y

	points := []image.Point{}
	for i := 0; i < len(l.Coords)-1; i += 2 {
		points = append(points, image.Point{
			X: minX + int(w*l.Coords[i]),
			Y: minY + int(h*l.Coords[i+1]),
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

func (l *Landmarks) DrawOnFullImage(cropped, full image.Image) image.Image {
	out := image.NewRGBA(full.Bounds())

	draw.Draw(out, full.Bounds(), full, image.ZP, draw.Src)

	for _, p := range l.PointsOnImage(cropped) {
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

func (l *Landmarks) Center(cropped, full image.Image) image.Image {
	onImage := l.PointsOnImage(cropped)

	noseTopX, noseTopY := float64(onImage[27].X), float64(onImage[27].Y)
	chinBottomX, chinBottomY := float64(onImage[8].X), float64(onImage[8].Y)
	dx := noseTopX - chinBottomX
	dy := noseTopY - chinBottomY
	var angle float64

	switch {
	case dx == 0:
		angle = 0
	case dy == 0 && dx > 0:
		angle = -90
	case dy == 0 && dx < 0:
		angle = 90
	default:
		angle = -math.Atan(dx/dy) * 180 / math.Pi
	}
	rotated := imaging.Rotate(full, angle, color.RGBA{A: 255})

	bounds := rotated.Bounds()
	minX, minY, maxX, maxY := bounds.Max.X, bounds.Max.Y, bounds.Min.X, bounds.Min.Y

	for _, p := range rotatePoints(angle,
		full.Bounds().Dx(), full.Bounds().Dy(),
		rotated.Bounds().Dx(), rotated.Bounds().Dy(), []image.Point{
			onImage[0],  // right chin top
			onImage[8],  // chin bottom
			onImage[16], // left chin top
			onImage[19], // right eyebrow top
			onImage[24], // left eyebrow top
		}) {
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
			X: int(float64(minX) - 10),
			Y: int(float64(minY) - 10),
		},
		Max: image.Point{
			X: int(float64(maxX) + 10),
			Y: int(float64(maxY) + 10),
		},
	}

	rect = square(rect)
	rect = insideOf(rect, bounds)

	out := image.NewRGBA(rect)

	draw.Draw(out, out.Bounds(), rotated, rect.Min, draw.Src)

	return out
}

func rotatePoints(angle float64, srcWidth, srcHeight, dstWidth, dstHeight int, points []image.Point) []image.Point {
	sin, cos := math.Sincos(-math.Pi * angle / 180)

	srcXOff := float64(srcWidth)/2 - 0.5
	srcYOff := float64(srcHeight)/2 - 0.5
	dstXOff := float64(dstWidth)/2 - 0.5
	dstYOff := float64(dstHeight)/2 - 0.5

	var rotated []image.Point
	for _, p := range points {
		x := float64(p.X) - srcXOff
		y := float64(p.Y) - srcYOff
		rotated = append(rotated, image.Point{
			X: int(x*cos - y*sin + dstXOff),
			Y: int(x*sin + y*cos + dstYOff),
		})
	}

	return rotated
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

func insideOf(rect, bounds image.Rectangle) image.Rectangle {
	if bounds.Min.X > rect.Min.X {
		rect.Max.X += bounds.Min.X - rect.Min.X
		rect.Min.X = bounds.Min.X
	}

	if bounds.Min.Y > rect.Min.Y {
		rect.Max.Y += bounds.Min.Y - rect.Min.Y
		rect.Min.Y = bounds.Min.Y
	}

	if rect.Max.X > bounds.Max.X {
		rect.Min.X -= rect.Max.X - bounds.Max.X
		rect.Max.X = bounds.Max.X
	}

	if rect.Max.Y > bounds.Max.Y {
		rect.Min.Y -= rect.Max.Y - bounds.Max.Y
		rect.Max.Y = bounds.Max.Y
	}

	return rect
}

type Descriptors []float32

func (d Descriptors) DistanceTo(d2 Descriptors) (float32, error) {
	if len(d) != len(d2) {
		return 0, errors.Errorf(
			"cannot calculate distance between descriptors of dimensions %d and %d", len(d), len(d2))
	}

	sum := float32(0)

	for i := 0; i < len(d); i++ {
		sum += (d[i] - d2[i]) * (d[i] - d2[i])
	}

	return float32(math.Sqrt(float64(sum))), nil
}

type Detector interface {
	Detect(img image.Image) ([]Detection, error)
}

type Landmark interface {
	Detect(img image.Image) (*Landmarks, error)
}

type Descriptor interface {
	Compute(img image.Image) (Descriptors, error)
}

type Extractor struct {
	Network    string
	Detector   Detector
	Landmark   Landmark
	Descriptor Descriptor
}

func (e *Extractor) Extract(img image.Image) ([]image.Image, []Descriptors, error) {
	_, _, _, _, centered, descrs, err := e.extract(img, 0.6, none)
	if err != nil {
		return nil, nil, err
	}

	return centered, descrs, nil
}

func (e *Extractor) ExtractLandmarks(img image.Image) ([][]image.Point, []image.Image, error) {
	_, cropped, _, landmarksOnImages, _, _, err := e.extract(img, 0.4, skipCenter)
	if err != nil {
		return nil, nil, err
	}

	return landmarksOnImages, cropped, nil
}

func (e *Extractor) ExtractPrimitives(img image.Image) ([]Detection, []Landmarks, []Descriptors, error) {
	detections, _, landmarks, _, _, descrs, err := e.extract(img, 0.6, none)
	if err != nil {
		return nil, nil, nil, err
	}

	return detections, landmarks, descrs, nil
}

const (
	none = iota
	skipCenter
	skipDescriptors
)

func (e *Extractor) extract(img image.Image, detectionThreshold float32, skip int) (
	detections []Detection,
	croppedCollection []image.Image,
	landmarksCollection []Landmarks,
	landmarksOnImages [][]image.Point,
	centeredCollection []image.Image,
	descriptorsCollection []Descriptors,
	err error) {
	allDetections, err := e.Detector.Detect(img)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, errors.Wrap(err, "error detecting faces")
	}

	detections = Above(allDetections, detectionThreshold)

	if len(detections) == 0 {
		return nil, nil, nil, nil, nil, nil, ErrNoFaceDetected
	}

	for _, d := range detections {
		if d.Box.Dx() < 45 || d.Box.Dy() < 45 {
			continue // face is too small
		}

		cropped := image.NewRGBA(d.Box)
		draw.Draw(cropped, d.Box, img, d.Box.Min, draw.Src)
		croppedCollection = append(croppedCollection, cropped)

		landmarks, err := e.Landmark.Detect(cropped)
		if err != nil {
			return nil, nil, nil, nil, nil, nil, errors.Wrap(err, "error detecting landmarks")
		}
		landmarksCollection = append(landmarksCollection, *landmarks)
		landmarksOnImages = append(landmarksOnImages, landmarks.PointsOnImage(cropped))

		if skip == skipCenter {
			continue
		}

		centered := landmarks.Center(cropped, img)
		centeredCollection = append(centeredCollection, centered)

		if skip == skipDescriptors {
			continue
		}

		descriptors, err := e.Descriptor.Compute(centered)
		if err != nil {
			return nil, nil, nil, nil, nil, nil, errors.Wrap(err, "error computing descriptors")
		}

		descriptorsCollection = append(descriptorsCollection, descriptors)
	}

	return detections,
		croppedCollection,
		landmarksCollection,
		landmarksOnImages,
		centeredCollection,
		descriptorsCollection, nil
}
