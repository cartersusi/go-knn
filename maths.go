package knn

import (
	"errors"
	"math"
)

func Einsum(qy []float64, db [][]float64) ([]float64, error) {
	if qy == nil || db == nil {
		return nil, errors.New("input slices must not be nil")
	}
	qCols := len(qy)
	dRows, dCols := len(db), len(db[0])

	if qCols != dCols {
		return nil, errors.New("input slices have the same number of columns")
	}

	result := make([]float64, dRows)

	for j := 0; j < dRows; j++ {
		sum := 0.0
		for k := 0; k < qCols; k++ {
			sum += qy[k] * db[j][k]
		}
		result[j] = sum
	}

	return result, nil
}

func HalfNorm(db [][]float64) []float64 {
	num_db := len(db)
	result := make([]float64, num_db)

	for i := 0; i < num_db; i++ {
		sum_squares := 0.0
		for j := 0; j < len(db[i]); j++ {
			sum_squares += db[i][j] * db[i][j]
		}
		result[i] = 0.5 * sum_squares
	}

	return result
}

func Manhattan(a, b []float64) float64 {
	var result float64
	for i := 0; i < len(a); i++ {
		result += math.Abs(a[i] - b[i])
	}

	return result
}
