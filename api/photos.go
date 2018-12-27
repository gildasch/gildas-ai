package api

import (
	"fmt"
	"net/http"
	"strings"

	gildasai "github.com/gildasch/gildas-ai"
	"github.com/gin-gonic/gin"
)

type Store interface {
	Contains(filename string) (bool, error)
	Get(query, after string, n int) ([]gildasai.PredictionItem, error)
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

func GetPhotoHandler(store Store) gin.HandlerFunc {
	return func(c *gin.Context) {
		filename := strings.Replace(c.Param("filename"), "//", "/", -1)

		ok, err := store.Contains(filename)
		if err != nil {
			fmt.Println(err)
			c.AbortWithStatus(http.StatusInternalServerError)
			return
		}
		if !ok {
			c.AbortWithStatus(http.StatusForbidden)
			return
		}

		c.File(filename)
	}
}
