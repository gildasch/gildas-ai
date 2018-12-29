package sqlite

import (
	"database/sql"
	"encoding/json"

	gildasai "github.com/gildasch/gildas-ai"
	_ "github.com/mattn/go-sqlite3"
	"github.com/pkg/errors"
)

type Store struct {
	*sql.DB
}

func NewStore(filename string) (*Store, error) {
	db, err := sql.Open("sqlite3", filename)
	if err != nil {
		return nil, err
	}

	createPredictionsDBStmt := `
create table if not exists predictions (
    id      text not null,
    network text not null,
    label   text not null,
    score   real not null,
    created timestamp default CURRENT_TIMESTAMP,
    primary key (id, network, label)
)
	`
	_, err = db.Exec(createPredictionsDBStmt)
	if err != nil {
		return nil, errors.Wrapf(err, "error running the SQL for DB creation %q\n", createPredictionsDBStmt)
	}

	createFaceDBStmt := `
create table if not exists faces (
    id          text not null,
    network     text not null,
    detection   text not null,
    landmarks   text not null,
    descriptors text not null,
    created     timestamp default CURRENT_TIMESTAMP,
    primary key (id, network, detection)
)
	`
	_, err = db.Exec(createFaceDBStmt)
	if err != nil {
		return nil, errors.Wrapf(err, "error running the SQL for DB creation %q\n", createFaceDBStmt)
	}

	return &Store{db}, nil
}

func (c *Store) GetPrediction(id string) (*gildasai.PredictionItem, bool, error) {
	rows, err := c.Query(`
select network, label, score
from predictions
where id = $1
order by score desc`, id)
	if err != nil {
		return nil, true, err
	}
	defer rows.Close()

	var preds gildasai.Predictions
	for rows.Next() {
		var network, label string
		var score float32
		err = rows.Scan(&network, &label, &score)
		if err != nil {
			return nil, true, err
		}
		preds = append(preds, gildasai.Prediction{
			Network: network,
			Label:   label,
			Score:   score,
		})
	}

	if len(preds) == 0 {
		return nil, false, nil
	}

	return &gildasai.PredictionItem{
		Identifier:  id,
		Predictions: preds}, true, nil
}

func (c *Store) StorePrediction(id string, item *gildasai.PredictionItem) error {
	tx, err := c.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	for _, p := range item.Predictions {
		_, err := tx.Exec(`
insert into predictions(id, network, label, score)
values ($1, $2, $3, $4)`,
			item.Identifier, p.Network, p.Label, p.Score)
		if err != nil {
			return err
		}
	}

	err = tx.Commit()
	if err != nil {
		return err
	}

	return nil
}

func (c *Store) SearchPrediction(query, after string, n int) ([]*gildasai.PredictionItem, error) {
	var rows *sql.Rows
	var err error

	if query != "" && after != "" {
		rows, err = c.Query(`
select id, network, label, score
from predictions
where id in (
  select id from predictions
  where label like $1 and id > $2
  limit $3
)
order by score desc`, "%"+query+"%", after, n)
	} else if query != "" {
		rows, err = c.Query(`
select id, network, label, score
from predictions
where id in (
  select id from predictions
  where label like $1
  limit $3
)
order by score desc`, "%"+query+"%", n)
	} else if after != "" {
		rows, err = c.Query(`
select distinct id, network, label, score
from predictions
where id > $2
order by score desc
limit $3`, after, n)
	} else {
		rows, err = c.Query(`
select distinct id, network, label, score
from predictions
order by score desc
limit $3`, n)
	}
	if err != nil {
		return nil, errors.Wrapf(err, "error querying sqlite store")
	}
	defer rows.Close()

	preds := map[string]gildasai.Predictions{}
	for rows.Next() {
		var id, networks, label string
		var score float32
		err := rows.Scan(&id, &networks, &label, &score)
		if err != nil {
			return nil, errors.Wrapf(err, "error scanning sqlite store")
		}

		preds[id] = append(preds[id], gildasai.Prediction{
			Network: networks,
			Label:   label,
			Score:   score})
	}

	var items []*gildasai.PredictionItem
	for id, p := range preds {
		items = append(items, &gildasai.PredictionItem{
			Identifier:  id,
			Predictions: p})
	}
	return items, nil
}

func (c *Store) StoreFace(item *gildasai.FaceItem) error {
	detection, err := json.Marshal(item.Detection)
	if err != nil {
		return nil
	}
	landmarks, err := json.Marshal(item.Landmarks)
	if err != nil {
		return nil
	}
	descriptors, err := json.Marshal(item.Descriptors)
	if err != nil {
		return nil
	}

	_, err = c.Exec(`
insert into faces(id, network, detection, landmarks, descriptors)
values ($1, $2, $3, $4, $5)`,
		item.Identifier, item.Network, string(detection), string(landmarks), string(descriptors))
	if err != nil {
		return err
	}

	return nil
}

func (c *Store) GetFaces() ([]*gildasai.FaceItem, error) {
	rows, err := c.Query(`
select id, network, detection, landmarks, descriptors
from faces`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var items []*gildasai.FaceItem
	for rows.Next() {
		var item gildasai.FaceItem
		var detection, landmarks, descriptors string
		err = rows.Scan(&item.Identifier, &item.Network, &detection, &landmarks, &descriptors)
		if err != nil {
			return nil, err
		}

		err = json.Unmarshal([]byte(detection), &item.Detection)
		if err != nil {
			return nil, err
		}
		err = json.Unmarshal([]byte(landmarks), &item.Landmarks)
		if err != nil {
			return nil, err
		}
		err = json.Unmarshal([]byte(descriptors), &item.Descriptors)
		if err != nil {
			return nil, err
		}

		items = append(items, &item)
	}

	return items, nil
}
