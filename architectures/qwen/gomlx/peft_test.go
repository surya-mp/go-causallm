//go:build gomlx

package gomlx

import (
	"math/rand"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/surya-mp/go-causallm/architectures/qwen"
	peftgomlx "github.com/surya-mp/go-peft/backends/gomlx"
	"github.com/surya-mp/go-peft/lora"
	"github.com/surya-mp/go-peft/qlora"
)

func TestModelImplementsGoPEFTLoRAHost(t *testing.T) {
	config, err := qwen.ParseConfig([]byte(`{"model_type":"qwen3","vocab_size":2,"hidden_size":2,"intermediate_size":4,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1}`))
	if err != nil {
		t.Fatal(err)
	}
	decoder, err := New(model.NewStore().RootScope().At("qwen"), config, dtypes.Float32)
	if err != nil {
		t.Fatal(err)
	}
	adapter, err := peftgomlx.Inject("adapter", decoder, lora.Config{
		Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"},
	}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	if len(adapter.Layers()) != 1 || len(decoder.overrides) != 1 {
		t.Fatalf("layers = %d, overrides = %d", len(adapter.Layers()), len(decoder.overrides))
	}
	weight, ok := decoder.Variable("model.layers.0.self_attn.q_proj.weight")
	if !ok || weight.Trainable {
		t.Fatal("base query projection must remain frozen")
	}
}

func TestInjectQLoRAReplacesBaseProjection(t *testing.T) {
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
	adapter, err := InjectQLoRA("adapter", decoder, qlora.Config{
		LoRA:      lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}},
		BlockSize: 2, Quantization: qlora.QuantizationNF4, DoubleQuant: true, ScaleBlockSize: 2,
	}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	if len(adapter.Layers()) != 1 || len(adapter.TrainableVariables()) != 2 {
		t.Fatalf("layers = %d, trainable = %d", len(adapter.Layers()), len(adapter.TrainableVariables()))
	}
	if _, exists := decoder.Variable("model.layers.0.self_attn.q_proj.weight"); exists {
		t.Fatal("full-precision query projection was retained after QLoRA injection")
	}
	if _, exists := decoder.Variable("model.layers.0.self_attn.k_proj.weight"); exists {
		t.Fatal("full-precision non-adapted projection was retained after QLoRA injection")
	}
	if _, exists := decoder.Variable("lm_head.weight"); exists {
		t.Fatal("full-precision language-model head was retained after QLoRA injection")
	}
	if len(decoder.overrides) != 8 {
		t.Fatalf("quantized overrides = %d, want 8", len(decoder.overrides))
	}
	exec, err := model.NewExec(gobackend.GetBackend(), store, func(_ *model.Scope, tokens *graph.Node) *graph.Node {
		return decoder.Logits(tokens, nil)
	})
	if err != nil {
		t.Fatal(err)
	}
	defer exec.Finalize()
	if _, err := exec.Exec1([][]int32{{0, 1}}); err != nil {
		t.Fatal(err)
	}
}
