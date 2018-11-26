package faces

import (
	"image"

	"github.com/gildasch/gildas-ai/faces/descriptors"
)

type Batch struct {
	Sources       map[string]image.Image
	Items         []*BatchItem
	Errors        []*BatchError
	Progress      Progress
	Notifications chan Progress
}

type BatchItem struct {
	Name        string
	Source      image.Image
	Cropped     image.Image
	Descriptors *descriptors.Descriptors
}

type BatchError struct {
	Name   string
	Source image.Image
	Error  error
}

type Progress struct {
	Count  int
	OK     int
	Errors int
}

func (b *Batch) Process(extractor *Extractor, jobs map[string]image.Image) *Batch {
	b.Sources = jobs
	for name, source := range jobs {
		item, err, progress := processOne(extractor, b.Progress, name, source)
		b.Items = append(b.Items, item...)
		b.Errors = append(b.Errors, err)
		b.Progress = progress
		if b.Notifications != nil {
			b.Notifications <- b.Progress
		}
	}

	return b
}

func processOne(extractor *Extractor, progress Progress, name string, source image.Image) ([]*BatchItem, *BatchError, Progress) {
	cropped, descrs, err := extractor.Extract(source)
	if err != nil {
		progress.Count++
		progress.Errors++
		return nil, &BatchError{
			Name:   name,
			Source: source,
			Error:  err,
		}, progress
	}

	ret := []*BatchItem{}
	for i := range descrs {
		ret = append(ret, &BatchItem{
			Name:        name,
			Source:      source,
			Cropped:     cropped[i],
			Descriptors: descrs[i],
		})
	}

	progress.Count++
	progress.OK++
	return ret, nil, progress
}

func (b *Batch) Distances() [][]float32 {
	progress := b.Progress

	distances := make([][]float32, progress.OK)

	for i := 0; i < progress.OK; i++ {
		distances[i] = make([]float32, progress.OK)
		for j := 0; j < progress.OK; j++ {
			distances[i][j] = b.Items[i].Descriptors.DistanceTo(b.Items[j].Descriptors)
		}
	}

	return distances
}
