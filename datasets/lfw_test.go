package datasets

import (
	"encoding/json"
	"fmt"
	"image/jpeg"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"

	gildasai "github.com/gildasch/gildas-ai"
	"github.com/gildasch/gildas-ai/faceapi"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	threshold           = 0.62
	minimumMatchTest    = 2000
	minimumNonMatchTest = 10000
)

func TestExtractionOnLFW(t *testing.T) {
	extractor, err := faceapi.NewDefaultExtractor("../faceapi")
	require.NoError(t, err)

	descrs, err := extract(extractor, "")
	require.NoError(t, err)

	actual, err := json.Marshal(descrs)
	require.NoError(t, err)

	expected, err := ioutil.ReadFile("lfw_temp-expected.json")
	require.NoError(t, err)

	assert.JSONEq(t, string(expected), string(actual))
}

func runEvaluation() bool {
	return os.Getenv("RUN_LFW_EVALUATION") == "1"
}

func TestLFWEvaluation(t *testing.T) {
	if !runEvaluation() {
		t.SkipNow()
	}

	extractor, err := faceapi.NewDefaultExtractor("../faceapi")
	require.NoError(t, err)

	descrs, err := extract(extractor, "lfw_temp.json")
	require.NoError(t, err)

	fmt.Println(len(descrs))

	var totalMatch, falseMatch, totalNonMatch, falseNonMatch int
evaluation:
	for name1, dd1 := range descrs {
		for _, d1 := range dd1 {
			for name2, dd2 := range descrs {
				for _, d2 := range dd2 {
					distance, err := d1.DistanceTo(d2)
					require.NoError(t, err)

					match := distance < threshold

					if name1 != name2 {
						totalNonMatch++
						if match {
							falseMatch++
						}
					}
					if name1 == name2 {
						totalMatch++
						if !match {
							falseNonMatch++
						}
					}
					fmt.Printf("\rtotal %d / false match %d (%.2f%%) / false non-match %d (%.2f%%)",
						totalMatch+totalNonMatch, falseMatch, (100 * float32(falseMatch) / float32(totalNonMatch)),
						falseNonMatch, (100 * float32(falseNonMatch) / float32(totalMatch)))

					if totalNonMatch >= minimumNonMatchTest && totalMatch >= minimumMatchTest {
						fmt.Printf("\nstopped at %d non-match tests and %d match test\n", totalNonMatch, totalMatch)
						break evaluation
					}
				}
			}
		}
	}
	fmt.Println()
}

func TestLFWEvaluationOnSubset(t *testing.T) {
	if !runEvaluation() {
		t.SkipNow()
	}

	extractor, err := faceapi.NewDefaultExtractor("../faceapi")
	require.NoError(t, err)

	descrs, err := extract(extractor, "lfw_temp.json")
	require.NoError(t, err)

	fmt.Println(len(descrs))

	var totalMatch, falseMatch, totalNonMatch, falseNonMatch int
	for _, name1 := range smallSetWithMultiplePictures {
		for _, d1 := range descrs[name1] {
			for _, name2 := range smallSetWithMultiplePictures {
				for _, d2 := range descrs[name2] {
					distance, err := d1.DistanceTo(d2)
					require.NoError(t, err)

					match := distance < threshold

					if name1 != name2 {
						totalNonMatch++
						if match {
							falseMatch++
						}
					}
					if name1 == name2 {
						totalMatch++
						if !match {
							falseNonMatch++
						}
					}
					fmt.Printf("\rtotal %d / false match %d (%.2f%%) / false non-match %d (%.2f%%)",
						totalMatch+totalNonMatch, falseMatch, (100 * float32(falseMatch) / float32(totalNonMatch)),
						falseNonMatch, (100 * float32(falseNonMatch) / float32(totalMatch)))
				}
			}
		}
	}
	fmt.Println()
}

func extract(extractor *gildasai.Extractor, saveFile string) (map[string][]gildasai.Descriptors, error) {
	descrs, ok := loadSaveFile(saveFile)
	if !ok {
		descrs = map[string][]gildasai.Descriptors{}
		fmt.Println("Extraction...")

		nameList, err := filepath.Glob("lfw/*")
		if err != nil {
			return nil, err
		}

		for n, name := range nameList {
			name = strings.TrimPrefix(name, "lfw/")
			filenames, err := filepath.Glob("lfw/" + name + "/*.jpg")
			if err != nil {
				return nil, err
			}

			for i, filename := range filenames {
				fmt.Printf("\rExtracting persone %d / image %d", n, i)

				f, err := os.Open(filename)
				if err != nil {
					return nil, err
				}

				img, err := jpeg.Decode(f)
				if err != nil {
					return nil, err
				}

				_, d, err := extractor.Extract(img)
				if len(d) >= 1 {
					descrs[name] = append(descrs[name], d[0])
				}
			}
		}
		fmt.Printf("\nExtraction done.\n")

		writeSaveFile(saveFile, descrs)
	}

	return descrs, nil
}

func loadSaveFile(saveFile string) (map[string][]gildasai.Descriptors, bool) {
	b, err := ioutil.ReadFile(saveFile)
	if err != nil {
		return nil, false
	}

	var descrs map[string][]gildasai.Descriptors
	err = json.Unmarshal(b, &descrs)
	if err != nil {
		return nil, false
	}

	return descrs, true
}

func writeSaveFile(saveFile string, descrs map[string][]gildasai.Descriptors) error {
	b, err := json.Marshal(descrs)
	if err != nil {
		return err
	}

	err = ioutil.WriteFile(saveFile, b, 0644)
	if err != nil {
		return err
	}

	return nil
}

var smallSetWithMultiplePictures = []string{
	"Jeb_Bush",
	"Joschka_Fischer",
	"Roh_Moo-hyun",
	"Kim_Clijsters",
	"Paul_Bremer",
	"Condoleezza_Rice",
	"John_Paul_II",
	"George_Robertson",
	"Bill_Clinton",
	"Ann_Veneman",
	"Wen_Jiabao",
	"Eduardo_Duhalde",
	"Tim_Henman",
	"Lindsay_Davenport",
	"Jose_Maria_Aznar",
	"Carlos_Menem",
	"Jeremy_Greenstock",
	"Spencer_Abraham",
	"Jiri_Novak",
	"Keanu_Reeves",
	"Walter_Mondale",
	"Bill_McBride",
	"Hugo_Chavez",
	"Silvio_Berlusconi",
	"Jean_Chretien",
	"Gloria_Macapagal_Arroyo",
	"John_Kerry",
	"Mahmoud_Abbas",
	"Tang_Jiaxuan",
	"Andy_Roddick",
	"George_HW_Bush",
	"Recep_Tayyip_Erdogan",
	"Taha_Yassin_Ramadan",
}
