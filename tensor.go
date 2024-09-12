package knn

import (
	"fmt"
	"reflect"
)

type Tensor[T float32 | float64] struct {
	Values interface{}
	Shape  [2]int
	Type   reflect.Type
	Rank   int
}

func (t *Tensor[T]) New(values interface{}) error {
	v := reflect.ValueOf(values)
	if v.Len() < 1 {
		return fmt.Errorf("empty values")
	}

	rank := 0
	var shape [2]int

	for v.Kind() == reflect.Slice {
		if rank >= 2 {
			return fmt.Errorf("unsupported rank: %d", rank+1)
		}
		shape[rank] = v.Len()
		v = v.Index(0)
		rank++
	}

	if shape[0] < 1 {
		return fmt.Errorf("empty values")
	}

	switch v.Kind() {
	case reflect.Float32, reflect.Float64:
		// Valid type
	default:
		return fmt.Errorf("unsupported type: %v", v.Type())
	}

	t.Values = values
	t.Shape = shape
	t.Rank = rank
	t.Type = v.Type()

	return nil
}
