// Package qwen implements the Qwen architecture plugin for causallm.
package qwen

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/surya-mp/go-causallm"
	basemodel "github.com/surya-mp/go-peft/format/huggingface/model"
)

var (
	// ErrInvalidConfig reports an unsupported or incomplete Qwen config.
	ErrInvalidConfig = errors.New("qwen: invalid configuration")
	// ErrNilTensorSink reports a missing backend tensor destination.
	ErrNilTensorSink = errors.New("qwen: nil tensor sink")
	// ErrMissingTensor reports a checkpoint missing one required base tensor.
	ErrMissingTensor = errors.New("qwen: missing checkpoint tensor")
	// ErrTensorShape reports a checkpoint tensor with an incompatible shape.
	ErrTensorShape = errors.New("qwen: invalid checkpoint tensor shape")
)

// Plugin maps Qwen checkpoint configuration and adapter targets to causallm.
type Plugin struct{}

// TensorSpec defines one required Hugging Face checkpoint tensor.
type TensorSpec struct {
	Name  string
	Shape []int
}

// TensorSink accepts one validated base tensor. Implementations should transfer
// or copy data before returning; the loader never retains a whole checkpoint.
type TensorSink interface {
	SetTensor(name, dtype string, shape []int, data []float32) error
}

// LoadOptions controls checkpoint loading diagnostics.
type LoadOptions struct {
	// Progress receives human-readable status messages during checkpoint load.
	Progress func(string)
	// ProgressEvery reports selected tensor progress every N tensors.
	// When zero, a modest default is used.
	ProgressEvery int
}

// Name identifies the Qwen architecture family.
func (Plugin) Name() string { return "qwen" }

// ParseConfig converts a Hugging Face Qwen config.json document into core config.
func ParseConfig(data []byte) (causallm.Config, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return causallm.Config{}, err
	}
	modelType, err := stringValue(raw, "model_type")
	if err != nil || (modelType != "qwen2" && modelType != "qwen3") {
		return causallm.Config{}, ErrInvalidConfig
	}
	values := map[string]any{"model_type": modelType}
	for _, key := range []string{
		"vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
		"num_attention_heads", "num_key_value_heads",
	} {
		value, err := intValue(raw, key)
		if err != nil {
			return causallm.Config{}, fmt.Errorf("%w: %s", ErrInvalidConfig, key)
		}
		values[key] = value
	}
	if value, exists, err := optionalIntValue(raw, "head_dim"); err != nil {
		return causallm.Config{}, err
	} else if exists {
		values["head_dim"] = value
	}
	for _, key := range []string{"rms_norm_eps", "rope_theta"} {
		if value, exists, err := floatValue(raw, key); err != nil {
			return causallm.Config{}, err
		} else if exists {
			values[key] = value
		}
	}
	for _, key := range []string{"tie_word_embeddings", "attention_bias", "mlp_bias"} {
		if value, exists, err := boolValue(raw, key); err != nil {
			return causallm.Config{}, err
		} else if exists {
			values[key] = value
		}
	}
	config := causallm.Config{Architecture: Plugin{}.Name(), Values: values}
	if err := (Plugin{}).Validate(config); err != nil {
		return causallm.Config{}, err
	}
	return config, nil
}

// Validate reports incompatible Qwen dimensions.
func (Plugin) Validate(config causallm.Config) error {
	if config.Architecture != (Plugin{}).Name() {
		return ErrInvalidConfig
	}
	modelType, ok := config.Values["model_type"].(string)
	if !ok || (modelType != "qwen2" && modelType != "qwen3") {
		return ErrInvalidConfig
	}
	hidden, hiddenOK := config.Values["hidden_size"].(int)
	heads, headsOK := config.Values["num_attention_heads"].(int)
	kvHeads, kvHeadsOK := config.Values["num_key_value_heads"].(int)
	if !hiddenOK || !headsOK || !kvHeadsOK || hidden <= 0 || heads <= 0 || kvHeads <= 0 ||
		heads%kvHeads != 0 {
		return ErrInvalidConfig
	}
	if headDim, exists := config.Values["head_dim"]; exists {
		value, ok := headDim.(int)
		if !ok || value <= 0 {
			return ErrInvalidConfig
		}
	} else if hidden%heads != 0 {
		return ErrInvalidConfig
	}
	for _, key := range []string{"vocab_size", "intermediate_size", "num_hidden_layers"} {
		value, ok := config.Values[key].(int)
		if !ok || value <= 0 {
			return ErrInvalidConfig
		}
	}
	return nil
}

// TargetModules returns Qwen linear projection suffixes suitable for LoRA.
func (Plugin) TargetModules(causallm.Config) ([]string, error) {
	return []string{"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}, nil
}

// ModuleName returns the canonical Hugging Face name for a Qwen linear target.
func ModuleName(layer int, target string) (string, error) {
	if layer < 0 {
		return "", ErrInvalidConfig
	}
	switch target {
	case "q_proj", "k_proj", "v_proj", "o_proj":
		return fmt.Sprintf("model.layers.%d.self_attn.%s", layer, target), nil
	case "gate_proj", "up_proj", "down_proj":
		return fmt.Sprintf("model.layers.%d.mlp.%s", layer, target), nil
	default:
		return "", fmt.Errorf("%w: target %q", ErrInvalidConfig, target)
	}
}

// TensorSpecs returns the required base-model checkpoint tensors in Hugging
// Face SafeTensors layout. Tied output embeddings are not listed twice.
func TensorSpecs(config causallm.Config) ([]TensorSpec, error) {
	if err := (Plugin{}).Validate(config); err != nil {
		return nil, err
	}
	vocab := config.Values["vocab_size"].(int)
	hidden := config.Values["hidden_size"].(int)
	intermediate := config.Values["intermediate_size"].(int)
	layers := config.Values["num_hidden_layers"].(int)
	heads := config.Values["num_attention_heads"].(int)
	kvHeads := config.Values["num_key_value_heads"].(int)
	headDim := headDimension(config)
	attentionBias, _ := config.Values["attention_bias"].(bool)
	mlpBias, _ := config.Values["mlp_bias"].(bool)
	tied, _ := config.Values["tie_word_embeddings"].(bool)
	specs := make([]TensorSpec, 0, 2+layers*(9+7))
	add := func(name string, shape ...int) {
		specs = append(specs, TensorSpec{Name: name, Shape: append([]int(nil), shape...)})
	}
	add("model.embed_tokens.weight", vocab, hidden)
	for layer := 0; layer < layers; layer++ {
		prefix := fmt.Sprintf("model.layers.%d", layer)
		add(prefix+".input_layernorm.weight", hidden)
		for _, projection := range []struct {
			name string
			out  int
			in   int
		}{
			{"q_proj", heads * headDim, hidden}, {"k_proj", kvHeads * headDim, hidden},
			{"v_proj", kvHeads * headDim, hidden}, {"o_proj", hidden, heads * headDim},
		} {
			add(prefix+".self_attn."+projection.name+".weight", projection.out, projection.in)
			if attentionBias {
				add(prefix+".self_attn."+projection.name+".bias", projection.out)
			}
		}
		add(prefix+".post_attention_layernorm.weight", hidden)
		for _, projection := range []struct {
			name string
			out  int
		}{
			{"gate_proj", intermediate}, {"up_proj", intermediate}, {"down_proj", hidden},
		} {
			input := hidden
			if projection.name == "down_proj" {
				input = intermediate
			}
			add(prefix+".mlp."+projection.name+".weight", projection.out, input)
			if mlpBias {
				add(prefix+".mlp."+projection.name+".bias", projection.out)
			}
		}
	}
	add("model.norm.weight", hidden)
	if !tied {
		add("lm_head.weight", vocab, hidden)
	}
	return specs, nil
}

// LoadSafeTensors streams a dense Qwen Hugging Face checkpoint into sink. It
// validates all required tensor names and shapes before reporting success.
func LoadSafeTensors(dir string, config causallm.Config, sink TensorSink) error {
	return LoadSafeTensorsWithOptions(dir, config, sink, LoadOptions{})
}

// LoadSafeTensorsWithOptions streams a dense Qwen Hugging Face checkpoint into
// sink with optional progress reporting.
func LoadSafeTensorsWithOptions(dir string, config causallm.Config, sink TensorSink, options LoadOptions) error {
	if sink == nil {
		return ErrNilTensorSink
	}
	specs, err := TensorSpecs(config)
	if err != nil {
		return err
	}
	progressf(options.Progress, "qwen: expecting %d tensors", len(specs))
	expected := make(map[string][]int, len(specs))
	for _, spec := range specs {
		expected[spec.Name] = spec.Shape
	}
	seen := make(map[string]struct{}, len(expected))
	progressEvery := options.ProgressEvery
	if progressEvery <= 0 {
		progressEvery = 25
	}
	_, err = basemodel.Load(dir, basemodel.Options{
		Filter: func(name string) bool {
			_, wanted := expected[name]
			return wanted
		},
		OnTensor: func(tensor basemodel.Tensor) error {
			next := len(seen) + 1
			if shouldReportTensor(next, len(expected), progressEvery) {
				progressf(options.Progress, "qwen: loading tensor %d/%d %s shape=%v dtype=%s shard=%s", next, len(expected), tensor.Name, tensor.Shape, tensor.DType, tensor.Shard)
			}
			shape := expected[tensor.Name]
			if !equalShape(shape, tensor.Shape) {
				return fmt.Errorf("%w: %s got %v want %v", ErrTensorShape, tensor.Name, tensor.Shape, shape)
			}
			if err := sink.SetTensor(tensor.Name, tensor.DType, tensor.Shape, tensor.Data); err != nil {
				return err
			}
			seen[tensor.Name] = struct{}{}
			return nil
		},
		Progress:      options.Progress,
		ProgressEvery: progressEvery,
	})
	if err != nil {
		return err
	}
	for _, spec := range specs {
		if _, exists := seen[spec.Name]; !exists {
			return fmt.Errorf("%w: %s", ErrMissingTensor, spec.Name)
		}
	}
	progressf(options.Progress, "qwen: loaded all %d tensors", len(seen))
	return nil
}

func progressf(progress func(string), format string, args ...any) {
	if progress != nil {
		progress(fmt.Sprintf(format, args...))
	}
}

func shouldReportTensor(index, total, every int) bool {
	return every > 0 && (index == 1 || index == total || index%every == 0)
}

func equalShape(left, right []int) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func stringValue(raw map[string]json.RawMessage, key string) (string, error) {
	value, exists := raw[key]
	if !exists {
		return "", ErrInvalidConfig
	}
	var result string
	if err := json.Unmarshal(value, &result); err != nil {
		return "", err
	}
	return result, nil
}

func intValue(raw map[string]json.RawMessage, key string) (int, error) {
	value, exists := raw[key]
	if !exists {
		return 0, ErrInvalidConfig
	}
	var result int
	if err := json.Unmarshal(value, &result); err != nil {
		return 0, err
	}
	return result, nil
}

func optionalIntValue(raw map[string]json.RawMessage, key string) (int, bool, error) {
	value, exists := raw[key]
	if !exists {
		return 0, false, nil
	}
	var result int
	if err := json.Unmarshal(value, &result); err != nil {
		return 0, false, err
	}
	return result, true, nil
}

func headDimension(config causallm.Config) int {
	if value, exists := config.Values["head_dim"].(int); exists {
		return value
	}
	return config.Values["hidden_size"].(int) / config.Values["num_attention_heads"].(int)
}

func floatValue(raw map[string]json.RawMessage, key string) (float64, bool, error) {
	value, exists := raw[key]
	if !exists {
		return 0, false, nil
	}
	var result float64
	if err := json.Unmarshal(value, &result); err != nil {
		return 0, false, err
	}
	return result, true, nil
}

func boolValue(raw map[string]json.RawMessage, key string) (bool, bool, error) {
	value, exists := raw[key]
	if !exists {
		return false, false, nil
	}
	var result bool
	if err := json.Unmarshal(value, &result); err != nil {
		return false, false, err
	}
	return result, true, nil
}
