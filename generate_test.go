package causallm

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

func TestGreedyGenerateStopsOnEOS(t *testing.T) {
	sequence := []int{4, 5, 2, 9}
	got, err := GreedyGenerate(context.Background(), []int{1}, func(_ context.Context, prefix []int) (int, error) {
		return sequence[len(prefix)-1], nil
	}, GenerateOptions{MaxNewTokens: 10, EOSID: 2, PADID: -1})
	if err != nil {
		t.Fatal(err)
	}
	want := []int{1, 4, 5, 2}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("tokens = %#v, want %#v", got, want)
	}
}

func TestGreedyGeneratePassesDefensivePrefix(t *testing.T) {
	prompt := []int{1}
	got, err := GreedyGenerate(context.Background(), prompt, func(_ context.Context, prefix []int) (int, error) {
		prefix[0] = 99
		return 2, nil
	}, GenerateOptions{MaxNewTokens: 1, EOSID: -1, PADID: -1})
	if err != nil {
		t.Fatal(err)
	}
	if prompt[0] != 1 || got[0] != 1 {
		t.Fatalf("prompt mutated: prompt=%#v got=%#v", prompt, got)
	}
}

func TestGreedyGenerateValidatesRequest(t *testing.T) {
	_, err := GreedyGenerate(context.Background(), nil, nil, GenerateOptions{})
	if !errors.Is(err, ErrInvalidGeneration) {
		t.Fatalf("err = %v", err)
	}
}

func TestGreedyGenerateHonorsContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := GreedyGenerate(ctx, []int{1}, func(context.Context, []int) (int, error) {
		return 2, nil
	}, GenerateOptions{MaxNewTokens: 1, EOSID: -1, PADID: -1})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
}
