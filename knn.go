package knn

import (
	"errors"
	"fmt"
)

type New struct {
	Data [][]float64
	K    int
}

type searchType string

const (
	L1   searchType = "L1"
	L2   searchType = "L2"
	MIPS searchType = "MIPS"
)

type SearchOptions interface{}
type BinSize struct {
	Value int
}
type RecallTarget struct {
	Value float64
}

func (s *New) Search(qy []float64, searchType searchType, opts ...SearchOptions) ([]int, []float64, error) {
	err := Validate(qy, s.Data, s.K)
	if err != nil {
		return nil, nil, err
	}

	switch searchType {
	case "L2":
		if len(opts) > 0 {
			l2nnsOpts, ok := opts[0].(RecallTarget)
			if !ok {
				return nil, nil, errors.New("invalid options for L2")
			}
			return L2nns(qy, s.Data, s.K, l2nnsOpts)
		}
		return L2nns(qy, s.Data, s.K)
	case "MIPS":
		if len(opts) > 0 {
			mipsOpts, ok := opts[0].(BinSize)
			if !ok {
				return nil, nil, errors.New("invalid options for MIPS")
			}
			return MIPSnns(qy, s.Data, s.K, mipsOpts)
		}
		return MIPSnns(qy, s.Data, s.K)
	case "L1":
		return L1nns(qy, s.Data, s.K)
	default:
		s.ListSearchTypes()
		return nil, nil, errors.New("search type not supported")
	}
}

func (s *New) ListSearchTypes() {
	fmt.Println("Available search types:")
	fmt.Println("\t\"L1\" - Perform L1-norm nearest neighbor search")
	fmt.Println("\t\"L2\" - Perform L2-norm nearest neighbor search")
	fmt.Println("\t\"MIPS\" - Perform MIPS search")
}

func Validate(qy []float64, db [][]float64, k int) error {
	if qy == nil || db == nil {
		return errors.New("input slices must not be nil")
	}
	if len(qy) == 0 || len(db) == 0 {
		return errors.New("input slices must not be empty")
	}
	if k <= 0 {
		return errors.New("k must be a positive integer")
	}
	if k > len(db) {
		return errors.New("k must be less than the length of the data")
	}
	return nil
}
