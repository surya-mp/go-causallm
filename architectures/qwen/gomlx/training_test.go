//go:build gomlx

package gomlx

import (
	"math/rand"
	"strings"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/optimizer"
	"github.com/surya-mp/go-causallm"
	"github.com/surya-mp/go-causallm/architectures/qwen"
	peftgomlx "github.com/surya-mp/go-peft/backends/gomlx"
	"github.com/surya-mp/go-peft/lora"
	"github.com/surya-mp/go-peft/qlora"
)

func TestQwenLoRATrainingStepUpdatesOnlyAdapter(t *testing.T) {
	config, err := qwen.ParseConfig([]byte(`{"model_type":"qwen3","vocab_size":4,"hidden_size":2,"intermediate_size":4,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1}`))
	if err != nil {
		t.Fatal(err)
	}
	store := model.NewStore()
	decoder, err := New(store.RootScope().At("qwen"), config, dtypes.Float32)
	if err != nil {
		t.Fatal(err)
	}
	fillQwen(t, decoder, config)
	adapter, err := peftgomlx.Inject("train", decoder, lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"o_proj"}}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	if len(adapter.TrainableVariables()) != 2 {
		t.Fatalf("trainable variables = %d", len(adapter.TrainableVariables()))
	}
	base, ok := decoder.Variable("model.layers.0.self_attn.o_proj.weight")
	if !ok || base.Trainable {
		t.Fatal("base output projection must be frozen")
	}
	baseBefore := variableData(t, base)
	before := variableData(t, adapter.TrainableVariables()[1])
	trainer := train.NewTrainer(gobackend.GetBackend(), store,
		func(_ *model.Scope, tokens *graph.Node) *graph.Node { return decoder.Logits(tokens, nil) },
		shiftedLoss, optimizer.Adam().LearningRate(0.1).Done(), nil, nil)
	batch := train.Batch{
		Inputs: []*tensors.Tensor{tensors.FromFlatDataAndDimensions([]int32{0, 1, 2}, 1, 3)},
		Labels: []*tensors.Tensor{tensors.FromFlatDataAndDimensions([]int32{-100, 1, 2}, 1, 3)},
	}
	if _, err := trainer.TrainStep(batch); err != nil {
		t.Fatal(err)
	}
	after := variableData(t, adapter.TrainableVariables()[1])
	if equalFloat32(before, after) {
		t.Fatal("LoRA B did not change after an optimizer step")
	}
	if got := variableData(t, base); !equalFloat32(baseBefore, got) {
		t.Fatal("frozen base projection changed after an optimizer step")
	}
}

func TestQwenQLoRATrainingStepUpdatesOnlyAdapter(t *testing.T) {
	config, err := qwen.ParseConfig([]byte(`{"model_type":"qwen3","vocab_size":4,"hidden_size":2,"intermediate_size":4,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1}`))
	if err != nil {
		t.Fatal(err)
	}
	store := model.NewStore()
	decoder, err := New(store.RootScope().At("qwen"), config, dtypes.Float32)
	if err != nil {
		t.Fatal(err)
	}
	fillQwen(t, decoder, config)
	adapter, err := InjectQLoRA("train", decoder, qlora.Config{
		LoRA: lora.Config{Rank: 1, Alpha: 1, TargetModules: []string{"o_proj"}}, BlockSize: 2,
		Quantization: qlora.QuantizationNF4, DoubleQuant: true, ScaleBlockSize: 2,
	}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	if len(adapter.TrainableVariables()) != 2 {
		t.Fatalf("trainable variables = %d", len(adapter.TrainableVariables()))
	}
	if _, exists := decoder.Variable("model.layers.0.self_attn.k_proj.weight"); exists {
		t.Fatal("QLoRA retained a full-precision frozen projection")
	}
	before := variableData(t, adapter.TrainableVariables()[1])
	trainer := train.NewTrainer(gobackend.GetBackend(), store,
		func(_ *model.Scope, tokens *graph.Node) *graph.Node { return decoder.Logits(tokens, nil) },
		shiftedLoss, optimizer.Adam().LearningRate(0.1).Done(), nil, nil)
	batch := train.Batch{
		Inputs: []*tensors.Tensor{tensors.FromFlatDataAndDimensions([]int32{0, 1, 2}, 1, 3)},
		Labels: []*tensors.Tensor{tensors.FromFlatDataAndDimensions([]int32{-100, 1, 2}, 1, 3)},
	}
	if _, err := trainer.TrainStep(batch); err != nil {
		t.Fatal(err)
	}
	if after := variableData(t, adapter.TrainableVariables()[1]); equalFloat32(before, after) {
		t.Fatal("QLoRA B did not change after an optimizer step")
	}
}

func fillQwen(t *testing.T, decoder *Model, config causallm.Config) {
	t.Helper()
	specs, err := qwen.TensorSpecs(config)
	if err != nil {
		t.Fatal(err)
	}
	for _, spec := range specs {
		size := 1
		for _, dimension := range spec.Shape {
			size *= dimension
		}
		value := float32(0.1)
		if strings.HasSuffix(spec.Name, "norm.weight") || strings.Contains(spec.Name, "layernorm.weight") {
			value = 1
		}
		data := make([]float32, size)
		for index := range data {
			data[index] = value
		}
		if err := decoder.SetTensor(spec.Name, "F32", spec.Shape, data); err != nil {
			t.Fatal(err)
		}
	}
}

func shiftedLoss(labels, predictions []*graph.Node) *graph.Node {
	nextLabels := graph.SliceAxis(labels[0], 1, graph.AxisRangeToEnd(1))
	nextLogits := graph.SliceAxis(predictions[0], 1, graph.AxisRangeFromStart(labels[0].Shape().Dimensions[1]-1))
	mask := graph.NotEqual(nextLabels, graph.Scalar(nextLabels.Graph(), nextLabels.DType(), int32(-100)))
	safeLabels := graph.Where(mask, nextLabels, graph.ZerosLike(nextLabels))
	return loss.SparseCategoricalCrossEntropyLogits([]*graph.Node{graph.InsertAxes(safeLabels, -1), mask}, []*graph.Node{nextLogits})
}

func variableData(t *testing.T, variable *model.Variable) []float32 {
	t.Helper()
	value, err := variable.Value()
	if err != nil {
		t.Fatal(err)
	}
	data, err := tensors.CopyFlatData[float32](value)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func equalFloat32(left, right []float32) bool {
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
