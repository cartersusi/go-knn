package knn

import (
	"errors"
	"math"
)

func Einsum(query []float64, database [][]float64) ([]float64, error) {
	if query == nil || database == nil {
		return nil, errors.New("input slices must not be nil")
	}
	qCols := len(query)
	dRows, dCols := len(database), len(database[0])

	if qCols != dCols {
		return nil, errors.New("query and database must have the same number of columns")
	}

	result := make([]float64, dRows)

	for j := 0; j < dRows; j++ {
		sum := 0.0
		for k := 0; k < qCols; k++ {
			sum += query[k] * database[j][k]
		}
		result[j] = sum
	}

	return result, nil
}

func HalfNorm(db [][]float64) []float64 {
	numDatabase := len(db)
	halfNorm := make([]float64, numDatabase)

	for i := 0; i < numDatabase; i++ {
		sumSquares := 0.0
		for j := 0; j < len(db[i]); j++ {
			sumSquares += db[i][j] * db[i][j]
		}
		halfNorm[i] = 0.5 * sumSquares
	}

	return halfNorm
}

func Euclidean(a, b []float64) (float64, error) {
	err := check_slices(a, b)
	if err != nil {
		return 0, err
	}

	var result float64
	for i := 0; i < len(a); i++ {
		result += math.Abs(a[i] - b[i])
	}

	return result, nil
}

func check_slices(a, b []float64) error {
	if a == nil || b == nil {
		return errors.New("input slices must not be nil")
	}
	if len(a) != len(b) {
		return errors.New("input slices must have the same length")
	}
	return nil
}
