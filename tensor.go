package knn

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"os"
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

func init() {
	gob.Register([]float32{})
	gob.Register([][]float32{})
	gob.Register([]float64{})
	gob.Register([][]float64{})
}

func (t *Tensor[T]) GobEncode() ([]byte, error) {
	var data struct {
		Values   interface{}
		Shape    [2]int
		TypeName string
		Rank     int
	}

	data.Values = t.Values
	data.Shape = t.Shape
	data.TypeName = t.Type.Name()
	data.Rank = t.Rank

	return gobEncode(data)
}

func (t *Tensor[T]) GobDecode(buf []byte) error {
	var data struct {
		Values   interface{}
		Shape    [2]int
		TypeName string
		Rank     int
	}

	if err := gobDecode(buf, &data); err != nil {
		return err
	}

	t.Values = data.Values
	t.Shape = data.Shape
	t.Rank = data.Rank

	switch data.TypeName {
	case "float32":
		t.Type = reflect.TypeOf(float32(0))
	case "float64":
		t.Type = reflect.TypeOf(float64(0))
	default:
		return fmt.Errorf("unsupported type: %s", data.TypeName)
	}

	return nil
}

func SaveTensor[T float32 | float64](t *Tensor[T], filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("error creating file: %v", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(t); err != nil {
		return fmt.Errorf("error encoding tensor: %v", err)
	}

	return nil
}

func LoadTensor[T float32 | float64](filename string) (*Tensor[T], error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("error opening file: %v", err)
	}
	defer file.Close()

	var t Tensor[T]
	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&t); err != nil {
		return nil, fmt.Errorf("error decoding tensor: %v", err)
	}

	return &t, nil
}

func gobEncode(data interface{}) ([]byte, error) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	err := enc.Encode(data)
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func gobDecode(buf []byte, data interface{}) error {
	dec := gob.NewDecoder(bytes.NewBuffer(buf))
	return dec.Decode(data)
}
