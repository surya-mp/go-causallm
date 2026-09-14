package qwen

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/surya-mp/go-causallm"
	"github.com/surya-mp/go-peft/format/safetensors"
)

func TestParseConfig(t *testing.T) {
	data := []byte("{\"model_type\":\"qwen3\",\"vocab_size\":10,\"hidden_size\":8,\"intermediate_size\":16,\"num_hidden_layers\":2,\"num_attention_heads\":2,\"num_key_value_heads\":1,\"rms_norm_eps\":0.000001}")
	config, err := ParseConfig(data)
	if err != nil {
		t.Fatal(err)
	}
	if config.Architecture != "qwen" || config.Values["hidden_size"] != 8 {
		t.Fatalf("config = %#v", config)
	}
	targets, err := (Plugin{}).TargetModules(config)
	if err != nil || !reflect.DeepEqual(targets[:4], []string{"q_proj", "k_proj", "v_proj", "o_proj"}) {
		t.Fatalf("targets = %v, err = %v", targets, err)
	}
}

func TestModuleName(t *testing.T) {
	name, err := ModuleName(1, "down_proj")
	if err != nil || name != "model.layers.1.mlp.down_proj" {
		t.Fatalf("name = %q, err = %v", name, err)
	}
}

func TestParseRejectsNonQwen(t *testing.T) {
	_, err := ParseConfig([]byte("{\"model_type\":\"llama\"}"))
	if !errors.Is(err, ErrInvalidConfig) {
		t.Fatalf("err = %v", err)
	}
}

func TestTensorSpecs(t *testing.T) {
	config, err := ParseConfig([]byte("{\"model_type\":\"qwen3\",\"vocab_size\":10,\"hidden_size\":8,\"intermediate_size\":16,\"num_hidden_layers\":2,\"num_attention_heads\":2,\"num_key_value_heads\":1}"))
	if err != nil {
		t.Fatal(err)
	}
	specs, err := TensorSpecs(config)
	if err != nil || len(specs) != 21 {
		t.Fatalf("spec count = %d, err = %v", len(specs), err)
	}
	if specs[0].Name != "model.embed_tokens.weight" || !reflect.DeepEqual(specs[0].Shape, []int{10, 8}) {
		t.Fatalf("embedding = %#v", specs[0])
	}
	var key TensorSpec
	for _, spec := range specs {
		if spec.Name == "model.layers.0.self_attn.k_proj.weight" {
			key = spec
			break
		}
	}
	if !reflect.DeepEqual(key.Shape, []int{4, 8}) {
		t.Fatalf("key projection = %#v", key)
	}
}

func TestTensorSpecsTiedAndBiased(t *testing.T) {
	config, err := ParseConfig([]byte("{\"model_type\":\"qwen2\",\"vocab_size\":10,\"hidden_size\":8,\"intermediate_size\":16,\"num_hidden_layers\":1,\"num_attention_heads\":2,\"num_key_value_heads\":1,\"tie_word_embeddings\":true,\"attention_bias\":true,\"mlp_bias\":true}"))
	if err != nil {
		t.Fatal(err)
	}
	specs, err := TensorSpecs(config)
	if err != nil || len(specs) != 18 {
		t.Fatalf("spec count = %d, err = %v", len(specs), err)
	}
	for _, spec := range specs {
		if spec.Name == "lm_head.weight" {
			t.Fatal("tied model must not require lm_head.weight")
		}
	}
}

func TestTensorSpecsUsesExplicitHeadDimension(t *testing.T) {
	config, err := ParseConfig([]byte("{\"model_type\":\"qwen3\",\"vocab_size\":10,\"hidden_size\":8,\"head_dim\":6,\"intermediate_size\":16,\"num_hidden_layers\":1,\"num_attention_heads\":2,\"num_key_value_heads\":1}"))
	if err != nil {
		t.Fatal(err)
	}
	specs, err := TensorSpecs(config)
	if err != nil {
		t.Fatal(err)
	}
	for _, spec := range specs {
		if spec.Name == "model.layers.0.self_attn.q_proj.weight" && !reflect.DeepEqual(spec.Shape, []int{12, 8}) {
			t.Fatalf("query shape = %v", spec.Shape)
		}
		if spec.Name == "model.layers.0.self_attn.o_proj.weight" && !reflect.DeepEqual(spec.Shape, []int{8, 12}) {
			t.Fatalf("output shape = %v", spec.Shape)
		}
	}
}

func TestLoadSafeTensors(t *testing.T) {
	config := tinyConfig(t)
	specs, err := TensorSpecs(config)
	if err != nil {
		t.Fatal(err)
	}
	dir := writeCheckpoint(t, specs, nil)
	sink := make(tensorSink)
	if err := LoadSafeTensors(dir, config, sink); err != nil {
		t.Fatal(err)
	}
	if len(sink) != len(specs) || sink["model.embed_tokens.weight"].dtype != "F32" {
		t.Fatalf("loaded = %#v", sink)
	}
}

func TestLoadSafeTensorsReportsProgress(t *testing.T) {
	config := tinyConfig(t)
	specs, err := TensorSpecs(config)
	if err != nil {
		t.Fatal(err)
	}
	dir := writeCheckpoint(t, specs, nil)
	var messages []string
	if err := LoadSafeTensorsWithOptions(dir, config, make(tensorSink), LoadOptions{
		Progress: func(message string) { messages = append(messages, message) },
	}); err != nil {
		t.Fatal(err)
	}
	if len(messages) < 3 {
		t.Fatalf("progress messages = %v", messages)
	}
	if !strings.HasPrefix(messages[0], "qwen: expecting ") {
		t.Fatalf("first progress message = %q", messages[0])
	}
}

func TestLoadSafeTensorsRejectsMissingAndWrongShape(t *testing.T) {
	config := tinyConfig(t)
	specs, err := TensorSpecs(config)
	if err != nil {
		t.Fatal(err)
	}
	dir := writeCheckpoint(t, specs[:len(specs)-1], nil)
	if err := LoadSafeTensors(dir, config, make(tensorSink)); !errors.Is(err, ErrMissingTensor) {
		t.Fatalf("missing err = %v", err)
	}
	dir = writeCheckpoint(t, specs, map[string][]int{"model.embed_tokens.weight": {1, 2}})
	if err := LoadSafeTensors(dir, config, make(tensorSink)); !errors.Is(err, ErrTensorShape) {
		t.Fatalf("shape err = %v", err)
	}
}

type capturedTensor struct {
	dtype string
	shape []int
}

type tensorSink map[string]capturedTensor

func (s tensorSink) SetTensor(name, dtype string, shape []int, _ []float32) error {
	s[name] = capturedTensor{dtype: dtype, shape: append([]int(nil), shape...)}
	return nil
}

func tinyConfig(t *testing.T) causallm.Config {
	t.Helper()
	config, err := ParseConfig([]byte("{\"model_type\":\"qwen3\",\"vocab_size\":2,\"hidden_size\":2,\"intermediate_size\":4,\"num_hidden_layers\":1,\"num_attention_heads\":1,\"num_key_value_heads\":1}"))
	if err != nil {
		t.Fatal(err)
	}
	return config
}

func writeCheckpoint(t *testing.T, specs []TensorSpec, shapes map[string][]int) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"qwen3"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	tensors := make(map[string]safetensors.Tensor, len(specs))
	for index, spec := range specs {
		shape := spec.Shape
		if override := shapes[spec.Name]; override != nil {
			shape = override
		}
		size := 1
		for _, dimension := range shape {
			size *= dimension
		}
		data := make([]float32, size)
		data[0] = float32(index + 1)
		tensors[spec.Name] = safetensors.Tensor{Shape: shape, Data: data}
	}
	if err := safetensors.WriteFile(filepath.Join(dir, "model.safetensors"), tensors, nil); err != nil {
		t.Fatal(err)
	}
	return dir
}
