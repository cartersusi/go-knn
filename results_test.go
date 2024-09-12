package knn

import (
	"container/heap"
	"reflect"
	"testing"
)

func TestMaxHeap(t *testing.T) {
	t.Run("Push and Pop", func(t *testing.T) {
		h := &MaxHeap[float32]{}
		heap.Init(h)

		items := []Result[float32]{
			{Index: 0, Distance: 5.0},
			{Index: 1, Distance: 3.0},
			{Index: 2, Distance: 7.0},
			{Index: 3, Distance: 1.0},
		}

		for _, item := range items {
			heap.Push(h, item)
		}

		expected := []Result[float32]{
			{Index: 2, Distance: 7.0},
			{Index: 0, Distance: 5.0},
			{Index: 1, Distance: 3.0},
			{Index: 3, Distance: 1.0},
		}

		for i := 0; i < len(items); i++ {
			result := heap.Pop(h).(Result[float32])
			if !reflect.DeepEqual(result, expected[i]) {
				t.Errorf("Pop %d: expected %v, got %v", i, expected[i], result)
			}
		}
	})

	t.Run("Peek", func(t *testing.T) {
		h := &MaxHeap[float64]{}
		heap.Init(h)

		items := []Result[float64]{
			{Index: 0, Distance: 5.0},
			{Index: 1, Distance: 3.0},
			{Index: 2, Distance: 7.0},
		}

		for _, item := range items {
			heap.Push(h, item)
		}

		expected := Result[float64]{Index: 2, Distance: 7.0}
		result := h.Peek().(Result[float64])

		if !reflect.DeepEqual(result, expected) {
			t.Errorf("Peek: expected %v, got %v", expected, result)
		}

		// Ensure Peek doesn't remove the item
		if h.Len() != 3 {
			t.Errorf("Peek should not remove items: expected length 3, got %d", h.Len())
		}
	})

	t.Run("Process", func(t *testing.T) {
		h := &MaxHeap[float32]{}
		heap.Init(h)

		k := 3
		items := []struct {
			index    int
			distance float32
		}{
			{0, 5.0},
			{1, 3.0},
			{2, 7.0},
			{3, 1.0},
			{4, 6.0},
			{5, 2.0},
		}

		for _, item := range items {
			h.Process(&item.index, &k, &item.distance)
		}

		expected := []Result[float32]{
			{Index: 1, Distance: 3.0},
			{Index: 5, Distance: 2.0},
			{Index: 3, Distance: 1.0},
		}

		if h.Len() != k {
			t.Errorf("Expected heap length %d, got %d", k, h.Len())
		}

		// Sort the heap contents for consistent comparison
		sortedResults := make([]Result[float32], k)
		for i := 0; i < k; i++ {
			sortedResults[i] = heap.Pop(h).(Result[float32])
		}

		if !reflect.DeepEqual(sortedResults, expected) {
			t.Errorf("Process results: expected %v, got %v", expected, sortedResults)
		}
	})
}

func BenchmarkMaxHeap(b *testing.B) {
	b.Run("Push and Pop", func(b *testing.B) {
		h := &MaxHeap[float64]{}
		heap.Init(h)

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			heap.Push(h, Result[float64]{Index: i, Distance: float64(i)})
			if h.Len() > 100 {
				heap.Pop(h)
			}
		}
	})

	b.Run("Process", func(b *testing.B) {
		h := &MaxHeap[float64]{}
		heap.Init(h)
		k := 100

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			index := i
			distance := float64(i)
			h.Process(&index, &k, &distance)
		}
	})
}
