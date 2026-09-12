// Package qwenmoe implements Qwen3-MoE checkpoint contracts.
package qwenmoe

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/surya-mp/go-causallm"
	basemodel "github.com/surya-mp/go-peft/format/huggingface/model"
)

var (
	ErrInvalidConfig = errors.New("qwenmoe: invalid configuration")
	ErrNilTensorSink = errors.New("qwenmoe: nil tensor sink")
	ErrMissingTensor = errors.New("qwenmoe: missing checkpoint tensor")
	ErrTensorShape   = errors.New("qwenmoe: invalid checkpoint tensor shape")
)

// Plugin maps Qwen3-MoE configuration and checkpoint names to causallm.
type Plugin struct{}

// TensorSpec defines one required Hugging Face checkpoint tensor.
type TensorSpec struct {
	Name  string
	Shape []int
}

// TensorSink accepts one validated checkpoint tensor.
type TensorSink interface {
	SetTensor(name, dtype string, shape []int, data []float32) error
}

// Name identifies the Qwen3-MoE architecture.
func (Plugin) Name() string { return "qwen3-moe" }

// ParseConfig converts a Hugging Face Qwen3-MoE config.json document.
func ParseConfig(data []byte) (causallm.Config, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return causallm.Config{}, err
	}
	var modelType string
	if err := json.Unmarshal(raw["model_type"], &modelType); err != nil || modelType != "qwen3_moe" {
		return causallm.Config{}, ErrInvalidConfig
	}
	values := map[string]any{"model_type": modelType}
	for _, key := range []string{"vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "moe_intermediate_size", "num_experts_per_tok", "decoder_sparse_step"} {
		value, err := requiredInt(raw, key)
		if err != nil {
			return causallm.Config{}, fmt.Errorf("%w: %s", ErrInvalidConfig, key)
		}
		values[key] = value
	}
	experts, err := optionalInt(raw, "num_experts")
	if err != nil {
		return causallm.Config{}, ErrInvalidConfig
	}
	if experts == 0 {
		experts, err = requiredInt(raw, "num_local_experts")
		if err != nil {
			return causallm.Config{}, fmt.Errorf("%w: num_experts", ErrInvalidConfig)
		}
	}
	values["num_experts"] = experts
	for _, key := range []string{"head_dim"} {
		if value, err := optionalInt(raw, key); err != nil {
			return causallm.Config{}, ErrInvalidConfig
		} else if value != 0 {
			values[key] = value
		}
	}
	for _, key := range []string{"rms_norm_eps", "rope_theta"} {
		if value, ok := raw[key]; ok {
			var number float64
			if json.Unmarshal(value, &number) != nil {
				return causallm.Config{}, ErrInvalidConfig
			}
			values[key] = number
		}
	}
	for _, key := range []string{"tie_word_embeddings", "attention_bias", "mlp_bias", "norm_topk_prob"} {
		if value, ok := raw[key]; ok {
			var flag bool
			if json.Unmarshal(value, &flag) != nil {
				return causallm.Config{}, ErrInvalidConfig
			}
			values[key] = flag
		}
	}
	if value, ok := raw["mlp_only_layers"]; ok {
		var layers []int
		if json.Unmarshal(value, &layers) != nil {
			return causallm.Config{}, ErrInvalidConfig
		}
		values["mlp_only_layers"] = layers
	}
	config := causallm.Config{Architecture: Plugin{}.Name(), Values: values}
	if err := (Plugin{}).Validate(config); err != nil {
		return causallm.Config{}, err
	}
	return config, nil
}

// Validate reports incompatible Qwen3-MoE dimensions.
func (Plugin) Validate(config causallm.Config) error {
	if config.Architecture != (Plugin{}).Name() || config.Values["model_type"] != "qwen3_moe" {
		return ErrInvalidConfig
	}
	for _, key := range []string{"vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "moe_intermediate_size", "num_experts", "num_experts_per_tok", "decoder_sparse_step"} {
		value, ok := config.Values[key].(int)
		if !ok || value <= 0 {
			return ErrInvalidConfig
		}
	}
	hidden := config.Values["hidden_size"].(int)
	heads := config.Values["num_attention_heads"].(int)
	kvHeads := config.Values["num_key_value_heads"].(int)
	if heads%kvHeads != 0 || (config.Values["num_experts_per_tok"].(int) > config.Values["num_experts"].(int)) {
		return ErrInvalidConfig
	}
	if headDim, exists := config.Values["head_dim"]; exists {
		if value, ok := headDim.(int); !ok || value <= 0 {
			return ErrInvalidConfig
		}
	} else if hidden%heads != 0 {
		return ErrInvalidConfig
	}
	if layers, exists := config.Values["mlp_only_layers"]; exists {
		list, ok := layers.([]int)
		if !ok {
			return ErrInvalidConfig
		}
		for _, layer := range list {
			if layer < 0 || layer >= config.Values["num_hidden_layers"].(int) {
				return ErrInvalidConfig
			}
		}
	}
	return nil
}

// TargetModules returns Qwen3-MoE adapter target suffixes.
func (Plugin) TargetModules(causallm.Config) ([]string, error) {
	return []string{"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "gate"}, nil
}

// SparseLayer reports whether layer contains routed experts.
func SparseLayer(config causallm.Config, layer int) bool {
	if (Plugin{}).Validate(config) != nil || layer < 0 || layer >= config.Values["num_hidden_layers"].(int) {
		return false
	}
	if layers, _ := config.Values["mlp_only_layers"].([]int); contains(layers, layer) {
		return false
	}
	return (layer+1)%config.Values["decoder_sparse_step"].(int) == 0
}

// TensorSpecs returns the Qwen3-MoE Hugging Face SafeTensors inventory.
func TensorSpecs(config causallm.Config) ([]TensorSpec, error) {
	if err := (Plugin{}).Validate(config); err != nil {
		return nil, err
	}
	vocab, hidden := config.Values["vocab_size"].(int), config.Values["hidden_size"].(int)
	intermediate, moeIntermediate := config.Values["intermediate_size"].(int), config.Values["moe_intermediate_size"].(int)
	layers, heads, kvHeads := config.Values["num_hidden_layers"].(int), config.Values["num_attention_heads"].(int), config.Values["num_key_value_heads"].(int)
	experts, headDim := config.Values["num_experts"].(int), headDimension(config)
	attentionBias, _ := config.Values["attention_bias"].(bool)
	mlpBias, _ := config.Values["mlp_bias"].(bool)
	tied, _ := config.Values["tie_word_embeddings"].(bool)
	specs := make([]TensorSpec, 0, 2+layers*(7+experts*3))
	add := func(name string, shape ...int) {
		specs = append(specs, TensorSpec{Name: name, Shape: append([]int(nil), shape...)})
	}
	add("model.embed_tokens.weight", vocab, hidden)
	for layer := 0; layer < layers; layer++ {
		prefix := fmt.Sprintf("model.layers.%d", layer)
		add(prefix+".input_layernorm.weight", hidden)
		for _, projection := range []struct {
			name    string
			out, in int
		}{
			{"q_proj", heads * headDim, hidden}, {"k_proj", kvHeads * headDim, hidden}, {"v_proj", kvHeads * headDim, hidden}, {"o_proj", hidden, heads * headDim},
		} {
			add(prefix+".self_attn."+projection.name+".weight", projection.out, projection.in)
			if attentionBias {
				add(prefix+".self_attn."+projection.name+".bias", projection.out)
			}
		}
		add(prefix+".post_attention_layernorm.weight", hidden)
		if SparseLayer(config, layer) {
			add(prefix+".mlp.gate.weight", experts, hidden)
			for expert := 0; expert < experts; expert++ {
				expertPrefix := fmt.Sprintf("%s.mlp.experts.%d", prefix, expert)
				add(expertPrefix+".gate_proj.weight", moeIntermediate, hidden)
				add(expertPrefix+".up_proj.weight", moeIntermediate, hidden)
				add(expertPrefix+".down_proj.weight", hidden, moeIntermediate)
			}
			continue
		}
		for _, projection := range []struct {
			name    string
			out, in int
		}{
			{"gate_proj", intermediate, hidden}, {"up_proj", intermediate, hidden}, {"down_proj", hidden, intermediate},
		} {
			add(prefix+".mlp."+projection.name+".weight", projection.out, projection.in)
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

// LoadSafeTensors streams a Qwen3-MoE checkpoint after name and shape validation.
func LoadSafeTensors(dir string, config causallm.Config, sink TensorSink) error {
	if sink == nil {
		return ErrNilTensorSink
	}
	specs, err := TensorSpecs(config)
	if err != nil {
		return err
	}
	expected, seen := make(map[string][]int, len(specs)), make(map[string]struct{}, len(specs))
	for _, spec := range specs {
		expected[spec.Name] = spec.Shape
	}
	_, err = basemodel.Load(dir, basemodel.Options{
		Filter: func(name string) bool { _, ok := expected[name]; return ok },
		OnTensor: func(tensor basemodel.Tensor) error {
			if !equalShape(expected[tensor.Name], tensor.Shape) {
				return fmt.Errorf("%w: %s", ErrTensorShape, tensor.Name)
			}
			if err := sink.SetTensor(tensor.Name, tensor.DType, tensor.Shape, tensor.Data); err != nil {
				return err
			}
			seen[tensor.Name] = struct{}{}
			return nil
		},
	})
	if err != nil {
		return err
	}
	for _, spec := range specs {
		if _, ok := seen[spec.Name]; !ok {
			return fmt.Errorf("%w: %s", ErrMissingTensor, spec.Name)
		}
	}
	return nil
}

func requiredInt(raw map[string]json.RawMessage, key string) (int, error) {
	value, ok := raw[key]
	if !ok {
		return 0, ErrInvalidConfig
	}
	var result int
	if err := json.Unmarshal(value, &result); err != nil {
		return 0, err
	}
	return result, nil
}
func optionalInt(raw map[string]json.RawMessage, key string) (int, error) {
	value, ok := raw[key]
	if !ok {
		return 0, nil
	}
	var result int
	if err := json.Unmarshal(value, &result); err != nil {
		return 0, err
	}
	return result, nil
}
func headDimension(config causallm.Config) int {
	if value, ok := config.Values["head_dim"].(int); ok {
		return value
	}
	return config.Values["hidden_size"].(int) / config.Values["num_attention_heads"].(int)
}
func contains(values []int, wanted int) bool {
	for _, value := range values {
		if value == wanted {
			return true
		}
	}
	return false
}
func equalShape(left, right []int) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}
