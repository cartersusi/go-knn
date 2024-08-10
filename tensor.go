package knn

import (
	"fmt"
	"reflect"
)

var Float = reflect.TypeOf(float32(0))
var Double = reflect.TypeOf(float64(0))

var validScalarTypes = map[reflect.Type]reflect.Type{
	reflect.TypeOf(float32(0)): reflect.TypeOf(float32(0)),
	reflect.TypeOf(float64(0)): reflect.TypeOf(float64(0)),
}

type Tensor struct {
	Values interface{}
	Shape  []int
	Type   reflect.Type
	Rank   int
}

func NewTensor(values interface{}) (Tensor, error) {
	var t Tensor

	v := reflect.ValueOf(values)

	rank := 0
	shape := []int{}

	for v.Kind() == reflect.Slice {
		shape = append(shape, v.Len())
		v = v.Index(0)
		rank++
	}

	if shape[0] < 1 {
		return t, fmt.Errorf("empty values")
	}

	if rank > 2 || rank < 1 {
		return t, fmt.Errorf("unsupported rank: %d", rank)
	}

	dtype, ok := validScalarTypes[v.Type()]
	if !ok {
		return t, fmt.Errorf("unsupported type: %v", v.Type())
	}

	t = Tensor{
		Values: values,
		Shape:  shape,
		Type:   dtype,
		Rank:   rank,
	}

	return t, nil
}

/* f32To64v()	f32To64m()	match_types()
* Floats will only be allowed to go from float32 to float64
* If both Tensor.Values are of different types, floats will be given full precision
* If both Tensor.Values are of the same type, no conversion will be done
 */

func (t *Tensor) ConvertToFloat64() {
	if t.Type == Float {
		switch values := t.Values.(type) {
		case []float32:
			t.Values = F32To64(values)
		case [][]float32:
			converted := make([][]float64, len(values))
			for i, v := range values {
				converted[i] = F32To64(v)
			}
			t.Values = converted
		default:
			fmt.Println("Unsupported tensor type")
		}
		t.Type = Double
	}
}

func F32To64(f32 []float32) []float64 {
	f64 := make([]float64, len(f32))
	for i, v := range f32 {
		f64[i] = float64(v)
	}
	return f64
}

func MatchTypes(query Tensor, data Tensor) (Tensor, Tensor) {
	if query.Type == Double && data.Type == Float {
		Log("Mismatching types. Converting to full precision...", Info)
		data.ConvertToFloat64()
	} else if query.Type == Float && data.Type == Double {
		Log("Mismatching types. Converting to full precision...", Info)
		query.ConvertToFloat64()
	}
	return query, data
}
