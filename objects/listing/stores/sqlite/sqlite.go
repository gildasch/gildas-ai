package sqlite

import (
	"database/sql"
	"strconv"
	"strings"

	gildasai "github.com/gildasch/gildas-ai"
	"github.com/gildasch/gildas-ai/objects/listing"
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

	createDBStmt := `
create table if not exists predictions (
    filename text not null,
    network text not null,
    label text not null,
    score real not null,
    created timestamp default CURRENT_TIMESTAMP,
    primary key (filename, network, label)
)
	`
	_, err = db.Exec(createDBStmt)
	if err != nil {
		return nil, errors.Wrapf(err, "error running the SQL for DB creation %q\n", createDBStmt)
	}

	return &Store{db}, nil
}

func (c *Store) Get(query, after string, n int) ([]listing.Item, error) {
	var rows *sql.Rows
	var err error

	if query != "" && after != "" {
		rows, err = c.Query(`
select filename, group_concat(label || ':' || score, ';')
from predictions
where label like $1 and filename > $2
group by filename
order by filename desc, score desc
limit $3`, "%"+query+"%", after, n)
	} else if query != "" {
		rows, err = c.Query(`
select filename, group_concat(label || ':' || score, ';')
from predictions
where label like $1
group by filename
order by filename desc, score desc
limit $3`, "%"+query+"%", n)
	} else if after != "" {
		rows, err = c.Query(`
select distinct filename, group_concat(label || ':' || score, ';')
from predictions
where filename > $2
group by filename
order by filename desc, score desc
limit $3`, after, n)
	} else {
		rows, err = c.Query(`
select distinct filename, group_concat(label || ':' || score, ';')
from predictions
group by filename
order by filename desc, score desc
limit $3`, n)
	}
	if err != nil {
		return nil, errors.Wrapf(err, "error querying sqlite store")
	}
	defer rows.Close()

	var items []listing.Item
	for rows.Next() {
		var filename, labelList string
		err := rows.Scan(&filename, &labelList)
		if err != nil {
			return nil, errors.Wrapf(err, "error scanning sqlite store")
		}

		items = append(items, listing.Item{
			Filename:    filename,
			Predictions: extractPredictions(labelList)})
	}

	return items, nil
}

func (c *Store) Contains(filename string) (bool, error) {
	rows, err := c.Query(`
select filename
from predictions
where filename = $1
limit 1`, filename)
	if err != nil {
		return false, err
	}
	defer rows.Close()

	return rows.Next(), nil
}

func extractPredictions(labelList string) []gildasai.Prediction {
	splitted := strings.Split(labelList, ";")

	var predictions []gildasai.Prediction
	for _, p := range splitted {
		labelAndScore := strings.Split(p, ":")
		if len(labelAndScore) != 2 {
			continue
		}

		label := labelAndScore[0]
		score, err := strconv.ParseFloat(labelAndScore[1], 32)
		if err != nil {
			continue
		}

		predictions = append(predictions, gildasai.Prediction{
			Label: label,
			Score: float32(score),
		})
	}

	return predictions
}
