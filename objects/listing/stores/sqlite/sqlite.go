package sqlite

import (
	"database/sql"

	"github.com/gildasch/gildas-ai/objects"
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
select filename, label, score
from predictions
where label = $1 and filename > $2
order by filename desc
limit $3`, query, after, n)
	} else if query != "" {
		rows, err = c.Query(`
select filename, label, score
from predictions
where label = $1
order by filename desc
limit $3`, query, n)
	} else if after != "" {
		rows, err = c.Query(`
select filename, label, score
from predictions
where filename > $2
order by filename desc
limit $3`, after, n)
	} else {
		rows, err = c.Query(`
select filename, label, score
from predictions
order by filename desc
limit $3`, n)
	}
	if err != nil {
		return nil, errors.Wrapf(err, "error querying sqlite store")
	}
	defer rows.Close()

	preds := map[string][]objects.Prediction{}
	for rows.Next() {
		var filename string
		var p objects.Prediction
		err := rows.Scan(&filename, &p.Label, &p.Score)
		if err != nil {
			return nil, errors.Wrapf(err, "error scanning sqlite store")
		}

		preds[filename] = append(preds[filename], p)
	}

	var items []listing.Item
	for filename, pp := range preds {
		items = append(items, listing.Item{
			Filename:    filename,
			Predictions: pp})
	}

	return items, nil
}
