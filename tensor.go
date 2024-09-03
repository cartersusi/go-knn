package knn

import (
	"fmt"
	"reflect"
)

type Tensor[T float32 | float64] struct {
	Values interface{}
	Shape  []int
	Type   reflect.Type
	Rank   int
}

var validScalarTypes = map[reflect.Type]reflect.Type{
	reflect.TypeOf(float32(0)): reflect.TypeOf(float32(0)),
	reflect.TypeOf(float64(0)): reflect.TypeOf(float64(0)),
}

func (t *Tensor[T]) New(values interface{}) error {
	v := reflect.ValueOf(values)

	rank := 0
	shape := []int{}

	for v.Kind() == reflect.Slice {
		shape = append(shape, v.Len())
		v = v.Index(0)
		rank++
	}

	if shape[0] < 1 {
		return fmt.Errorf("empty values")
	}

	if rank > 2 || rank < 1 {
		return fmt.Errorf("unsupported rank: %d", rank)
	}

	dtype, ok := validScalarTypes[v.Type()]
	if !ok {
		return fmt.Errorf("unsupported type: %v", v.Type())
	}

	if rank == 1 {
		t.Values = values.([]T)
	} else {
		t.Values = values.([][]T)
	}

	t.Shape = shape
	t.Rank = rank
	t.Type = dtype

	return nil
}
