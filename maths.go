package knn

import (
	"math"
)

func Einsumf64(qy []float64, db [][]float64) []float64 {
	qCols := len(qy)
	dRows := len(db)

	result := make([]float64, dRows)

	for j := 0; j < dRows; j++ {
		sum := 0.0
		for k := 0; k < qCols; k++ {
			sum += qy[k] * db[j][k]
		}
		result[j] = sum
	}

	return result
}

func Einsumf32(qy []float32, db [][]float32) []float32 {
	qCols := len(qy)
	dRows := len(db)

	result := make([]float32, dRows)

	var sum float32
	for j := 0; j < dRows; j++ {
		sum = 0.0
		for k := 0; k < qCols; k++ {
			sum += qy[k] * db[j][k]
		}
		result[j] = sum
	}

	return result
}

func HalfNormf64(db [][]float64) []float64 {
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

func HalfNormf32(db [][]float32) []float64 {
	num_db := len(db)
	result := make([]float64, num_db)

	var sum_squares float32
	for i := 0; i < num_db; i++ {
		sum_squares = 0.0
		for j := 0; j < len(db[i]); j++ {
			sum_squares += db[i][j] * db[i][j]
		}
		result[i] = 0.5 * float64(sum_squares)
	}

	return result
}

func Manhattanf64(a, b []float64) float64 {
	var result float64
	for i := 0; i < len(a); i++ {
		result += math.Abs(a[i] - b[i])
	}

	return result
}

func Manhattanf32(a, b []float32) float64 {
	var result float64
	for i := 0; i < len(a); i++ {
		result += math.Abs(float64(a[i] - b[i]))
	}

	return result
}
