package knn

import (
	"errors"
	"fmt"
)

type New struct {
	Data Tensor
	K    int
}

const (
	L1 = iota
	L2
	MIPS
)

func (s *New) Search(query Tensor, searchType int, opts ...interface{}) ([]int, []float64, error) {
	query, s.Data = MatchTypes(query, s.Data)

	err := Validate(query, s.Data, s.K)
	if err != nil {
		return nil, nil, err
	}

	switch searchType {
	case L2:
		l2nnsOpts := 0.95
		if len(opts) > 0 {
			switch v := opts[0].(type) {
			case float64:
				l2nnsOpts = v
			case float32:
				l2nnsOpts = float64(v)
			default:
				return nil, nil, errors.New("invalid options for L2")
			}
		}
		return L2nns(query, s.Data, s.K, l2nnsOpts)
	case MIPS:
		mipsOpts := EstimateBinSize(s.Data.Shape[0])
		if len(opts) > 0 {
			mipsOpts, ok := opts[0].(int)
			if !ok {
				return nil, nil, errors.New("invalid options for MIPS")
			}
			return MIPSnns(query, s.Data, s.K, mipsOpts)
		}
		return MIPSnns(query, s.Data, s.K, mipsOpts)
	case L1:
		return L1nns(query, s.Data, s.K)
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

func Validate(qy Tensor, db Tensor, k int) error {
	if qy.Values == nil || db.Values == nil {
		return errors.New("input slices must not be nil")
	}
	if qy.Shape[0] == 0 || db.Shape[0] == 0 || db.Shape[1] == 0 {
		return errors.New("input slices must not be empty")
	}
	if qy.Shape[0] != db.Shape[1] {
		return errors.New("query and data slices must have the same length")
	}
	if k <= 0 {
		return errors.New("k must be a positive integer")
	}
	if k > db.Shape[0] {
		return errors.New("k must be less than the length of the data")
	}
	return nil
}
