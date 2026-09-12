package qwenmoe

import (
	"strings"
	"testing"
)

func TestParseConfigAndTensorSpecs(t *testing.T) {
	config, err := ParseConfig([]byte(`{
"model_type":"qwen3_moe","vocab_size":8,"hidden_size":4,"intermediate_size":6,
"num_hidden_layers":3,"num_attention_heads":2,"num_key_value_heads":1,
"moe_intermediate_size":3,"num_experts":2,"num_experts_per_tok":1,"decoder_sparse_step":2,
"mlp_only_layers":[2],"rms_norm_eps":0.000001,"rope_theta":10000
}`))
	if err != nil || !SparseLayer(config, 1) || SparseLayer(config, 0) || SparseLayer(config, 2) {
		t.Fatalf("config = %#v, err = %v", config, err)
	}
	specs, err := TensorSpecs(config)
	if err != nil {
		t.Fatal(err)
	}
	names := make([]string, len(specs))
	for i := range specs {
		names[i] = specs[i].Name
	}
	joined := strings.Join(names, ",")
	for _, name := range []string{
		"model.layers.0.mlp.gate_proj.weight", "model.layers.1.mlp.gate.weight",
		"model.layers.1.mlp.experts.1.down_proj.weight", "model.layers.2.mlp.down_proj.weight",
	} {
		if !strings.Contains(joined, name) {
			t.Fatalf("missing %s", name)
		}
	}
}

func TestParseConfigRejectsInvalidTopK(t *testing.T) {
	_, err := ParseConfig([]byte(`{"model_type":"qwen3_moe","vocab_size":8,"hidden_size":4,"intermediate_size":6,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"moe_intermediate_size":3,"num_experts":1,"num_experts_per_tok":2,"decoder_sparse_step":1}`))
	if err == nil {
		t.Fatal("expected validation error")
	}
}
