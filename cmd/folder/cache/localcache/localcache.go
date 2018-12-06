package localcache

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
)

type LocalCache struct {
	CacheFile string

	inceptions map[string][]string
	saved      int
}

func (l *LocalCache) Inception(file string, inception func() ([]string, error)) ([]string, error) {
	if l.inceptions == nil {
		var err error
		l.inceptions, err = readCache(l.CacheFile)
		if err == nil {
			l.saved = len(l.inceptions)
		} else {
			fmt.Println("no cache found")
			l.inceptions = map[string][]string{}
			l.saved = 0
		}
	}

	if preds, ok := l.inceptions[file]; ok {
		fmt.Printf("loaded file %q from cache\n", file)
		return preds, nil
	}

	preds, err := inception()
	if err != nil {
		return nil, err
	}

	l.inceptions[file] = preds

	if len(l.inceptions) > l.saved+10 {
		err := saveCache(l.CacheFile, l.inceptions)
		if err != nil {
			fmt.Println("error saving cache:", err)
		} else {
			l.saved = len(l.inceptions)
		}
	}

	return preds, nil
}

func readCache(cacheFile string) (map[string][]string, error) {
	b, err := ioutil.ReadFile(cacheFile)
	if err != nil {
		return nil, err
	}

	var inceptions map[string][]string
	err = json.Unmarshal(b, &inceptions)
	if err != nil {
		return nil, err
	}

	return inceptions, nil
}

func saveCache(cacheFile string, inceptions map[string][]string) error {
	b, err := json.Marshal(inceptions)
	if err != nil {
		return err
	}

	err = ioutil.WriteFile(cacheFile, b, 0644)
	if err != nil {
		return err
	}

	return nil
}
