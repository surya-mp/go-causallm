package gomlx

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/surya-mp/go-causallm"
	"github.com/surya-mp/go-causallm/architectures/qwen"
	"github.com/surya-mp/go-peft/format/safetensors"
)

func TestDenseQwenBuildsForwardGraph(t *testing.T) {
	config, err := qwen.ParseConfig([]byte(`{"model_type":"qwen3","vocab_size":2,"hidden_size":2,"intermediate_size":4,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1}`))
	if err != nil {
		t.Fatal(err)
	}
	store := model.NewStore()
	decoder, err := New(store.RootScope().At("qwen"), config, dtypes.Float32)
	if err != nil {
		t.Fatal(err)
	}
	dir := writeCheckpoint(t, config)
	if err := qwen.LoadSafeTensors(dir, config, decoder); err != nil {
		t.Fatal(err)
	}
	exec, err := model.NewExec(gobackend.GetBackend(), store, func(_ *model.Scope, tokens *graph.Node) *graph.Node {
		return decoder.Logits(tokens, nil)
	})
	if err != nil {
		t.Fatal(err)
	}
	defer exec.Finalize()
	logits, err := exec.Exec1([][]int32{{0, 1}})
	if err != nil {
		t.Fatal(err)
	}
	if got := logits.Shape().Dimensions; len(got) != 3 || got[0] != 1 || got[1] != 2 || got[2] != 2 {
		t.Fatalf("logits shape = %v", got)
	}
}

func TestDenseQwenBuildsPackedForwardGraph(t *testing.T) {
	config, err := qwen.ParseConfig([]byte(`{"model_type":"qwen3","vocab_size":2,"hidden_size":2,"intermediate_size":4,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1}`))
	if err != nil {
		t.Fatal(err)
	}
	store := model.NewStore()
	decoder, err := New(store.RootScope().At("qwen"), config, dtypes.Float32)
	if err != nil {
		t.Fatal(err)
	}
	if err := qwen.LoadSafeTensors(writeCheckpoint(t, config), config, decoder); err != nil {
		t.Fatal(err)
	}
	exec, err := model.NewExec(gobackend.GetBackend(), store, func(_ *model.Scope, tokens, segments *graph.Node) *graph.Node {
		return decoder.LogitsSegmented(tokens, segments)
	})
	if err != nil {
		t.Fatal(err)
	}
	defer exec.Finalize()
	logits, err := exec.Exec1([][]int32{{0, 1, 0}}, [][]int32{{0, 0, 1}})
	if err != nil || logits.Shape().Dimensions[1] != 3 {
		t.Fatalf("logits = %v, err = %v", logits.Shape(), err)
	}
}

func writeCheckpoint(t *testing.T, config causallm.Config) string {
	t.Helper()
	specs, err := qwen.TensorSpecs(config)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"qwen3"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	tensors := make(map[string]safetensors.Tensor, len(specs))
	for _, spec := range specs {
		size := 1
		for _, dimension := range spec.Shape {
			size *= dimension
		}
		tensors[spec.Name] = safetensors.Tensor{Shape: spec.Shape, Data: make([]float32, size)}
	}
	if err := safetensors.WriteFile(filepath.Join(dir, "model.safetensors"), tensors, nil); err != nil {
		t.Fatal(err)
	}
	return dir
}
