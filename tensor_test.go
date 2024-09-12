package knn

import (
	"reflect"
	"testing"
)

func TestTensorNew(t *testing.T) {
	tests := []struct {
		name    string
		input   interface{}
		want    Tensor[float32]
		wantErr bool
	}{
		{
			name:  "Valid 1D float32 slice",
			input: []float32{1.0, 2.0, 3.0},
			want: Tensor[float32]{
				Values: []float32{1.0, 2.0, 3.0},
				Shape:  [2]int{3, 0},
				Type:   reflect.TypeOf(float32(0)),
				Rank:   1,
			},
			wantErr: false,
		},
		{
			name:  "Valid 2D float32 slice",
			input: [][]float32{{1.0, 2.0}, {3.0, 4.0}},
			want: Tensor[float32]{
				Values: [][]float32{{1.0, 2.0}, {3.0, 4.0}},
				Shape:  [2]int{2, 2},
				Type:   reflect.TypeOf(float32(0)),
				Rank:   2,
			},
			wantErr: false,
		},
		{
			name:    "Invalid empty 1D slice",
			input:   []float32{},
			wantErr: true,
		},
		{
			name:    "Invalid empty 2D slice",
			input:   [][]float32{},
			wantErr: true,
		},
		{
			name:    "Invalid 3D slice",
			input:   [][][]float32{{{1.0}}},
			wantErr: true,
		},
		{
			name:    "Invalid type (int)",
			input:   []int{1, 2, 3},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := &Tensor[float32]{}
			err := tensor.New(tt.input)

			if (err != nil) != tt.wantErr {
				t.Errorf("Tensor.New() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if !reflect.DeepEqual(tensor.Values, tt.want.Values) {
					t.Errorf("Tensor.New() Values = %v, want %v", tensor.Values, tt.want.Values)
				}
				if !reflect.DeepEqual(tensor.Shape, tt.want.Shape) {
					t.Errorf("Tensor.New() Shape = %v, want %v", tensor.Shape, tt.want.Shape)
				}
				if tensor.Type != tt.want.Type {
					t.Errorf("Tensor.New() Type = %v, want %v", tensor.Type, tt.want.Type)
				}
				if tensor.Rank != tt.want.Rank {
					t.Errorf("Tensor.New() Rank = %v, want %v", tensor.Rank, tt.want.Rank)
				}
			}
		})
	}
}

func TestTensorNewFloat64(t *testing.T) {
	tests := []struct {
		name    string
		input   interface{}
		want    Tensor[float64]
		wantErr bool
	}{
		{
			name:  "Valid 1D float64 slice",
			input: []float64{1.0, 2.0, 3.0},
			want: Tensor[float64]{
				Values: []float64{1.0, 2.0, 3.0},
				Shape:  [2]int{3, 0},
				Type:   reflect.TypeOf(float64(0)),
				Rank:   1,
			},
			wantErr: false,
		},
		{
			name:  "Valid 2D float64 slice",
			input: [][]float64{{1.0, 2.0}, {3.0, 4.0}},
			want: Tensor[float64]{
				Values: [][]float64{{1.0, 2.0}, {3.0, 4.0}},
				Shape:  [2]int{2, 2},
				Type:   reflect.TypeOf(float64(0)),
				Rank:   2,
			},
			wantErr: false,
		},
		{
			name:    "Invalid empty 3D slice",
			input:   [][][]float64{{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}}, {{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}}},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := &Tensor[float64]{}
			err := tensor.New(tt.input)

			if (err != nil) != tt.wantErr {
				t.Errorf("Tensor.New() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {
				if !reflect.DeepEqual(tensor.Values, tt.want.Values) {
					t.Errorf("Tensor.New() Values = %v, want %v", tensor.Values, tt.want.Values)
				}
				if !reflect.DeepEqual(tensor.Shape, tt.want.Shape) {
					t.Errorf("Tensor.New() Shape = %v, want %v", tensor.Shape, tt.want.Shape)
				}
				if tensor.Type != tt.want.Type {
					t.Errorf("Tensor.New() Type = %v, want %v", tensor.Type, tt.want.Type)
				}
				if tensor.Rank != tt.want.Rank {
					t.Errorf("Tensor.New() Rank = %v, want %v", tensor.Rank, tt.want.Rank)
				}
			}
		})
	}
}

func BenchmarkTensorNew(b *testing.B) {
	benchmarks := []struct {
		name  string
		input interface{}
	}{
		{
			name:  "1D float32 slice (small)",
			input: make([]float32, 100),
		},
		{
			name:  "1D float32 slice (medium)",
			input: make([]float32, 10000),
		},
		{
			name:  "1D float32 slice (large)",
			input: make([]float32, 1000000),
		},
		{
			name:  "2D float32 slice (small)",
			input: make([][]float32, 10, 10),
		},
		{
			name:  "2D float32 slice (medium)",
			input: make([][]float32, 100, 100),
		},
		{
			name:  "2D float32 slice (large)",
			input: make([][]float32, 1000, 1000),
		},
		{
			name:  "1D float64 slice (small)",
			input: make([]float64, 100),
		},
		{
			name:  "1D float64 slice (medium)",
			input: make([]float64, 10000),
		},
		{
			name:  "1D float64 slice (large)",
			input: make([]float64, 1000000),
		},
		{
			name:  "2D float64 slice (small)",
			input: make([][]float64, 10, 10),
		},
		{
			name:  "2D float64 slice (medium)",
			input: make([][]float64, 100, 100),
		},
		{
			name:  "2D float64 slice (large)",
			input: make([][]float64, 1000, 1000),
		},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				switch v := bm.input.(type) {
				case []float32:
					tensor := &Tensor[float32]{}
					_ = tensor.New(v)
				case [][]float32:
					tensor := &Tensor[float32]{}
					for j := range v {
						v[j] = make([]float32, len(v))
					}
					_ = tensor.New(v)
				case []float64:
					tensor := &Tensor[float64]{}
					_ = tensor.New(v)
				case [][]float64:
					tensor := &Tensor[float64]{}
					for j := range v {
						v[j] = make([]float64, len(v))
					}
					_ = tensor.New(v)
				}
			}
		})
	}
}
