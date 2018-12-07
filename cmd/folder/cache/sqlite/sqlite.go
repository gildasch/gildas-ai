package sqlite

import (
	"database/sql"
	"fmt"

	"github.com/gildasch/gildas-ai/tensor"
	_ "github.com/mattn/go-sqlite3"
	"github.com/pkg/errors"
)

type Cache struct {
	*sql.DB
}

func NewCache(filename string) (*Cache, error) {
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

	return &Cache{db}, nil
}

func (c *Cache) Inception(file, network string, inception func() ([]tensor.Prediction, error)) ([]tensor.Prediction, error) {
	rows, err := c.Query("select label, score from predictions where filename=$1 and network=$2", file, network)
	if err != nil {
		return nil, errors.Wrapf(err, "error querying sqlite cache for %q / %q", file, network)
	}
	defer rows.Close()

	var preds []tensor.Prediction
	for rows.Next() {
		var p tensor.Prediction
		err := rows.Scan(&p.Label, &p.Score)
		if err != nil {
			return nil, errors.Wrapf(err, "error scanning sqlite cache for %q / %q", file, network)
		}
		preds = append(preds, p)
	}

	if len(preds) > 0 {
		fmt.Printf("predictions for %q / %q loaded from sqlite cache\n", file, network)
		return preds, nil
	}

	preds, err = inception()
	if err != nil {
		return nil, errors.Wrapf(err, "error running inception on %q / %q", file, network)
	}

	tx, err := c.Begin()
	if err != nil {
		return nil, errors.Wrap(err, "cannot create sqlite transaction")
	}
	defer tx.Rollback()

	for _, p := range preds {
		_, err := tx.Exec("insert into predictions (filename, network, label, score) values ($1, $2, $3, $4)",
			file, network, p.Label, p.Score)
		if err != nil {
			return nil, errors.Wrap(err, "error inserting new row in sqlite DB")
		}
	}

	err = tx.Commit()
	if err != nil {
		return nil, errors.Wrap(err, "error commiting the sqlite transaction")
	}

	return preds, nil
}
