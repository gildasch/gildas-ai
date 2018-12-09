package labels

import (
	"encoding/json"
	"io/ioutil"
	"strconv"

	"github.com/pkg/errors"
)

type Labels map[int]LabelItem

type LabelItem struct {
	Name, Description string
}

func FromFile(jsonFile string, indexCorrection int) (Labels, error) {
	j, err := ioutil.ReadFile(jsonFile)
	if err != nil {
		return nil, errors.Wrapf(err, "could not read file %q", jsonFile)
	}

	var l map[string][2]string
	err = json.Unmarshal(j, &l)
	if err != nil {
		return nil, errors.Wrap(err, "json unmarshalling failed")
	}

	labels := make(map[int]LabelItem)

	for i, ll := range l {
		ii, err := strconv.Atoi(i)
		if err != nil {
			return nil, errors.Wrapf(err, "the labels file has a bad index %q", i)
		}

		ii -= indexCorrection

		labels[ii] = LabelItem{
			Name:        ll[0],
			Description: ll[1],
		}
	}

	return labels, nil
}

func (l Labels) Get(i int) string {
	if ll, ok := l[i]; ok {
		return ll.Description
	}
	return strconv.Itoa(i)
}
