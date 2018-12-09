package api

import (
	"fmt"
	"net/http"

	"github.com/gildasch/gildas-ai/objects/listing"
	"github.com/gin-gonic/gin"
)

type Store interface {
	Get(query, after string, n int) ([]listing.Item, error)
}

func PhotosHandler(store Store) gin.HandlerFunc {
	return func(c *gin.Context) {
		query := c.Query("query")
		after := c.Query("after")
		n := 100

		items, err := store.Get(query, after, n)
		if err != nil {
			fmt.Println(err)
			c.AbortWithStatus(http.StatusInternalServerError)
			return
		}

		c.HTML(http.StatusOK, "photos.html", gin.H{
			"query": query,
			"after": after,
			"items": items,
		})

	}
}

func GetPhotoHandler(c *gin.Context) {
	c.File(c.Param("filename"))
}
