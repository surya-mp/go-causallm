//go:build gomlx

package gomlx

import (
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/surya-mp/go-causallm/architectures/qwen"
)

func TestQwenGeneratorConfiguresKVCache(t *testing.T) {
	config, err := qwen.ParseConfig([]byte(tinyQwenConfigJSON))
	if err != nil {
		t.Fatal(err)
	}
	session, err := LoadSession(writeCheckpoint(t, config), SessionOptions{DType: dtypes.Float32})
	if err != nil {
		t.Fatal(err)
	}
	cache := session.Model.KVCache()
	if len(cache.OrderedScopes) != 1 {
		t.Fatalf("cache scopes = %#v", cache.OrderedScopes)
	}
	generator := session.Model.Generator().WithMaxLength(4).WithEOS(-1)
	if generator == nil {
		t.Fatal("generator is nil")
	}
}
