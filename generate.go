package causallm

import (
	"context"
	"errors"
)

var ErrInvalidGeneration = errors.New("causallm: invalid generation request")

// NextTokenFunc returns the next token for the current prompt plus generated prefix.
type NextTokenFunc func(context.Context, []int) (int, error)

// GenerateOptions controls greedy autoregressive decoding.
type GenerateOptions struct {
	MaxNewTokens int
	EOSID        int
	PADID        int
	StopIDs      []int
}

// GreedyGenerate appends argmax-selected tokens until max-new-tokens or a stop
// token is reached. It is backend-neutral; model packages provide NextTokenFunc.
func GreedyGenerate(ctx context.Context, prompt []int, next NextTokenFunc, options GenerateOptions) ([]int, error) {
	if len(prompt) == 0 || next == nil || options.MaxNewTokens <= 0 {
		return nil, ErrInvalidGeneration
	}
	stop := make(map[int]struct{}, len(options.StopIDs)+2)
	if options.EOSID >= 0 {
		stop[options.EOSID] = struct{}{}
	}
	if options.PADID >= 0 {
		stop[options.PADID] = struct{}{}
	}
	for _, id := range options.StopIDs {
		stop[id] = struct{}{}
	}
	generated := append([]int(nil), prompt...)
	for i := 0; i < options.MaxNewTokens; i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		token, err := next(ctx, append([]int(nil), generated...))
		if err != nil {
			return nil, err
		}
		generated = append(generated, token)
		if _, ok := stop[token]; ok {
			break
		}
	}
	return generated, nil
}
