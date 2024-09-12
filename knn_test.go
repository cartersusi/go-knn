package knn

import (
	"math"
	"reflect"
	"testing"
)

func TestSearch(t *testing.T) {
	// Test data
	data := [][]float32{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
		{10.0, 11.0, 12.0},
	}
	query := []float32{3.0, 4.0, 5.0}

	// Create tensors
	dataTensor := &Tensor[float32]{}
	err := dataTensor.New(data)
	if err != nil {
		t.Fatalf("Failed to create data tensor: %v", err)
	}

	queryTensor := &Tensor[float32]{}
	err = queryTensor.New(query)
	if err != nil {
		t.Fatalf("Failed to create query tensor: %v", err)
	}

	// Create search struct
	s := &Search[float32]{
		Data:  dataTensor,
		Query: queryTensor,
	}

	// Test L1 (Manhattan) distance
	t.Run("L1 Distance", func(t *testing.T) {
		neighbors, err := s.L1(2)
		if err != nil {
			t.Fatalf("L1 search failed: %v", err)
		}

		expectedIndices := []int{1, 0}
		expectedValues := []float32{3, 6}

		if !reflect.DeepEqual(neighbors.Indices, expectedIndices) {
			t.Errorf("L1 indices mismatch. Got %v, want %v", neighbors.Indices, expectedIndices)
		}

		for i, v := range neighbors.Values {
			if math.Abs(float64(v-expectedValues[i])) > 1e-6 {
				t.Errorf("L1 value mismatch at index %d. Got %f, want %f", i, v, expectedValues[i])
			}
		}
	})

	// Test L2 (Euclidean) distance
	t.Run("L2 Distance", func(t *testing.T) {
		neighbors, err := s.L2(2)
		if err != nil {
			t.Fatalf("L2 search failed: %v", err)
		}

		expectedIndices := []int{1, 0}
		expectedValues := []float32{-23.5, -19.0}

		if !reflect.DeepEqual(neighbors.Indices, expectedIndices) {
			t.Errorf("L2 indices mismatch. Got %v, want %v", neighbors.Indices, expectedIndices)
		}

		for i, v := range neighbors.Values {
			if math.Abs(float64(v-expectedValues[i])) > 1e-6 {
				t.Errorf("L2 value mismatch at index %d. Got %f, want %f", i, v, expectedValues[i])
			}
		}
	})

	// Test MIPS (Maximum Inner Product Search)
	t.Run("MIPS", func(t *testing.T) {
		neighbors, err := s.MIPS(2)
		if err != nil {
			t.Fatalf("MIPS search failed: %v", err)
		}

		expectedIndices := []int{3, 2}
		expectedValues := []float32{134, 98}

		if !reflect.DeepEqual(neighbors.Indices, expectedIndices) {
			t.Errorf("MIPS indices mismatch. Got %v, want %v", neighbors.Indices, expectedIndices)
		}

		for i, v := range neighbors.Values {
			if math.Abs(float64(v-expectedValues[i])) > 1e-6 {
				t.Errorf("MIPS value mismatch at index %d. Got %f, want %f", i, v, expectedValues[i])
			}
		}
	})

	// Test error cases
	t.Run("Error Cases", func(t *testing.T) {
		// Test with invalid k
		_, err := s.L1(0)
		if err == nil {
			t.Error("Expected error for k=0, got nil")
		}

		_, err = s.L1(len(data) + 1)
		if err == nil {
			t.Error("Expected error for k > data length, got nil")
		}

		// Test with mismatched dimensions
		invalidQuery := []float32{1.0, 2.0} // Different dimension from data
		invalidQueryTensor := &Tensor[float32]{}
		_ = invalidQueryTensor.New(invalidQuery)

		invalidSearch := &Search[float32]{
			Data:  dataTensor,
			Query: invalidQueryTensor,
		}

		_, err = invalidSearch.L1(1)
		if err == nil {
			t.Error("Expected error for mismatched dimensions, got nil")
		}
	})
}

func BenchmarkSearch(b *testing.B) {
	// Prepare larger dataset for benchmarking
	data := make([][]float32, 10000)
	for i := range data {
		data[i] = make([]float32, 128)
		for j := range data[i] {
			data[i][j] = float32(i*128 + j)
		}
	}
	query := make([]float32, 128)
	for i := range query {
		query[i] = float32(i)
	}

	dataTensor := &Tensor[float32]{}
	_ = dataTensor.New(data)
	queryTensor := &Tensor[float32]{}
	_ = queryTensor.New(query)

	s := &Search[float32]{
		Data:  dataTensor,
		Query: queryTensor,
	}

	b.Run("L1", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = s.L1(10)
		}
	})

	b.Run("L2", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = s.L2(10)
		}
	})

	b.Run("MIPS", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = s.MIPS(10)
		}
	})
}

func BenchmarkLargeMultithreadedSearch(b *testing.B) {
	// Prepare larger dataset for benchmarking
	data := make([][]float32, 10000)
	for i := range data {
		data[i] = make([]float32, 128)
		for j := range data[i] {
			data[i][j] = float32(i*128 + j)
		}
	}
	query := make([]float32, 128)
	for i := range query {
		query[i] = float32(i)
	}

	dataTensor := &Tensor[float32]{}
	_ = dataTensor.New(data)
	queryTensor := &Tensor[float32]{}
	_ = queryTensor.New(query)

	s := &Search[float32]{
		Data:        dataTensor,
		Query:       queryTensor,
		Multithread: true,
		MaxWorkers:  8,
	}

	b.Run("L1", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = s.L1(10)
		}
	})

	b.Run("L2", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = s.L2(10)
		}
	})

	/* No multithreaded MIPS implementation
	b.Run("MIPS", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = s.MIPS(10)
		}
	})*/
}
