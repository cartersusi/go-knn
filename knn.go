package knn

import (
	"errors"
	"fmt"
)

type New struct {
	Data [][]float64
	K    int
}

type SearchOptions interface{}

type MipsOptions struct {
	BinSize int
}

type L2nnsOptions struct {
	RecallTarget float64
}

func (s *New) Search(qy []float64, searchType string, opts ...SearchOptions) ([]int, []float64, error) {
	switch searchType {
	case "L2":
		if len(opts) > 0 {
			l2nnsOpts, ok := opts[0].(L2nnsOptions)
			if !ok {
				return nil, nil, errors.New("invalid options for L2nns")
			}
			return L2nns(qy, s.Data, s.K, l2nnsOpts)
		}
		return L2nns(qy, s.Data, s.K)
	case "MIPS":
		if len(opts) > 0 {
			mipsOpts, ok := opts[0].(MipsOptions)
			if !ok {
				return nil, nil, errors.New("invalid options for MIPS")
			}
			return MIPS(qy, s.Data, s.K, mipsOpts)
		}
		return MIPS(qy, s.Data, s.K)
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
