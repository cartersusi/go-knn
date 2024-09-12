package knn

import (
	"math"
	"reflect"
	"runtime"
	"testing"
)

func TestManhattan(t *testing.T) {
	tests := []struct {
		name     string
		data     [][]float32
		query    []float32
		index    int
		expected float32
	}{
		{
			name:     "Basic test",
			data:     [][]float32{{1, 2, 3}, {4, 5, 6}},
			query:    []float32{1, 1, 1},
			index:    0,
			expected: 3,
		},
		{
			name:     "Zero vector",
			data:     [][]float32{{0, 0, 0}, {1, 1, 1}},
			query:    []float32{0, 0, 0},
			index:    1,
			expected: 3,
		},
		{
			name:     "Negative values",
			data:     [][]float32{{-1, -2, -3}, {1, 2, 3}},
			query:    []float32{0, 0, 0},
			index:    0,
			expected: 6,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &Search[float32]{
				Data:  &Tensor[float32]{Values: tt.data},
				Query: &Tensor[float32]{Values: tt.query},
			}
			result := s.Manhattan(&tt.index)
			if result != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestEinsum(t *testing.T) {
	tests := []struct {
		name        string
		data        [][]float32
		query       []float32
		expected    []float32
		multithread bool
	}{
		{
			name:        "Basic test",
			data:        [][]float32{{1, 2, 3}, {4, 5, 6}},
			query:       []float32{1, 1, 1},
			expected:    []float32{6, 15},
			multithread: false,
		},
		{
			name:        "Zero vector",
			data:        [][]float32{{1, 2, 3}, {4, 5, 6}},
			query:       []float32{0, 0, 0},
			expected:    []float32{0, 0},
			multithread: false,
		},
		{
			name:        "Multithreaded",
			data:        [][]float32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
			query:       []float32{1, 1, 1},
			expected:    []float32{6, 15, 24},
			multithread: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &Search[float32]{
				Data:        &Tensor[float32]{Values: tt.data, Shape: [2]int{len(tt.data), len(tt.data[0])}},
				Query:       &Tensor[float32]{Values: tt.query, Shape: [2]int{len(tt.query)}},
				Multithread: tt.multithread,
				MaxWorkers:  runtime.NumCPU(),
			}
			result := s.Einsum()
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestHalfNorm(t *testing.T) {
	tests := []struct {
		name        string
		data        [][]float32
		expected    []float32
		multithread bool
	}{
		{
			name:        "Basic test",
			data:        [][]float32{{1, 2, 3}, {4, 5, 6}},
			expected:    []float32{7, 38.5},
			multithread: false,
		},
		{
			name:        "Zero vector",
			data:        [][]float32{{0, 0, 0}, {1, 1, 1}},
			expected:    []float32{0, 1.5},
			multithread: false,
		},
		{
			name:        "Multithreaded",
			data:        [][]float32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
			expected:    []float32{7, 38.5, 97},
			multithread: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &Search[float32]{
				Data:        &Tensor[float32]{Values: tt.data, Shape: [2]int{len(tt.data), len(tt.data[0])}},
				Multithread: tt.multithread,
				MaxWorkers:  runtime.NumCPU(),
			}
			result := s.HalfNorm()
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestEstimateBinSize(t *testing.T) {
	tests := []struct {
		name     string
		dataSize int
		expected int
	}{
		{"Small data", 100, 1},
		{"Medium data", 5000, 4},
		{"Large data", 100000, 8},
		{"Very large data", 300000000, 64},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &Search[float32]{
				Data: &Tensor[float32]{Shape: [2]int{tt.dataSize, 10}},
			}
			result := s.EstimateBinSize()
			if result != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func BenchmarkManhattan(b *testing.B) {
	data := make([][]float32, 1000)
	for i := range data {
		data[i] = make([]float32, 100)
		for j := range data[i] {
			data[i][j] = float32(i + j)
		}
	}
	query := make([]float32, 100)
	for i := range query {
		query[i] = float32(i)
	}

	s := &Search[float32]{
		Data:  &Tensor[float32]{Values: data},
		Query: &Tensor[float32]{Values: query},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		index := i % 1000
		_ = s.Manhattan(&index)
	}
}

func BenchmarkEinsum(b *testing.B) {
	data := make([][]float32, 1000)
	for i := range data {
		data[i] = make([]float32, 100)
		for j := range data[i] {
			data[i][j] = float32(i + j)
		}
	}
	query := make([]float32, 100)
	for i := range query {
		query[i] = float32(i)
	}

	benchmarks := []struct {
		name        string
		multithread bool
	}{
		{"SingleThreaded", false},
		{"MultiThreaded", true},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			s := &Search[float32]{
				Data:        &Tensor[float32]{Values: data, Shape: [2]int{1000, 100}},
				Query:       &Tensor[float32]{Values: query, Shape: [2]int{100}},
				Multithread: bm.multithread,
				MaxWorkers:  runtime.NumCPU(),
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = s.Einsum()
			}
		})
	}
}

func BenchmarkHalfNorm(b *testing.B) {
	data := make([][]float32, 1000)
	for i := range data {
		data[i] = make([]float32, 100)
		for j := range data[i] {
			data[i][j] = float32(i + j)
		}
	}

	benchmarks := []struct {
		name        string
		multithread bool
	}{
		{"SingleThreaded", false},
		{"MultiThreaded", true},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			s := &Search[float32]{
				Data:        &Tensor[float32]{Values: data, Shape: [2]int{1000, 100}},
				Multithread: bm.multithread,
				MaxWorkers:  runtime.NumCPU(),
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = s.HalfNorm()
			}
		})
	}
}

func BenchmarkEstimateBinSize(b *testing.B) {
	s := &Search[float32]{
		Data: &Tensor[float32]{Shape: [2]int{1000000, 100}},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = s.EstimateBinSize()
	}
}

func almostEqual(a, b float32, epsilon float32) bool {
	return math.Abs(float64(a-b)) <= float64(epsilon)
}
