// Package gomlx builds dense Qwen graphs with GoMLX variables named after
// Hugging Face checkpoint tensors.
package gomlx

import (
	"errors"
	"fmt"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/layers/attention"
	"github.com/gomlx/gomlx/ml/layers/attention/pos"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/nn"
	"github.com/gomlx/gomlx/support/exceptions"
	"github.com/surya-mp/go-causallm"
	"github.com/surya-mp/go-causallm/architectures/qwen"
)

var (
	// ErrInvalidModel reports a missing scope or invalid graph configuration.
	ErrInvalidModel = errors.New("qwen/gomlx: invalid model")
	// ErrUnsupportedDType reports a variable dtype this bridge cannot load.
	ErrUnsupportedDType = errors.New("qwen/gomlx: unsupported dtype")
	// ErrUnknownTensor reports a tensor not owned by the dense Qwen graph.
	ErrUnknownTensor = errors.New("qwen/gomlx: unknown tensor")
)

// Model is a dense Qwen decoder graph backed by named GoMLX variables.
type Model struct {
	scope     *model.Scope
	config    causallm.Config
	dtype     dtypes.DType
	variables map[string]*model.Variable
	overrides map[string]LinearOverride
	quantized map[string]any
}

// LinearOverride replaces one named base linear projection.
type LinearOverride interface {
	Apply(scope *model.Scope, input *graph.Node) *graph.Node
}

// New creates frozen base-model variables matching the Qwen checkpoint layout.
func New(scope *model.Scope, config causallm.Config, dtype dtypes.DType) (*Model, error) {
	if scope == nil || !supportedDType(dtype) {
		return nil, ErrInvalidModel
	}
	specs, err := qwen.TensorSpecs(config)
	if err != nil {
		return nil, err
	}
	m := &Model{
		scope: scope, config: config, dtype: dtype,
		variables: make(map[string]*model.Variable, len(specs)), overrides: make(map[string]LinearOverride),
		quantized: make(map[string]any),
	}
	for _, spec := range specs {
		variableScope, variableName := scopeForTensor(scope, spec.Name)
		variable := variableScope.VariableWithShape(variableName, shape(dtype, spec.Shape...))
		variable.Trainable = false
		m.variables[spec.Name] = variable
	}
	return m, nil
}

// SetTensor implements qwen.TensorSink and assigns one validated checkpoint tensor.
func (m *Model) SetTensor(name, _ string, dimensions []int, data []float32) error {
	if m == nil {
		return ErrInvalidModel
	}
	variable, exists := m.variables[name]
	if !exists {
		return fmt.Errorf("%w: %s", ErrUnknownTensor, name)
	}
	if !equalDimensions(variable.Shape().Dimensions, dimensions) {
		return fmt.Errorf("%w: %s", qwen.ErrTensorShape, name)
	}
	value, err := tensorFromFloat32(m.dtype, data, dimensions)
	if err != nil {
		return err
	}
	return variable.SetValue(value)
}

// Variable returns the GoMLX variable associated with one checkpoint tensor.
func (m *Model) Variable(name string) (*model.Variable, bool) {
	variable, exists := m.variables[name]
	return variable, exists
}

// Logits builds dense Qwen causal-LM logits for tokens [B, T]. sequenceLengths
// is optional int32 [B] metadata for right-padded batches.
func (m *Model) Logits(tokens, sequenceLengths *graph.Node) *graph.Node {
	if m == nil || tokens == nil || tokens.Rank() != 2 || !tokens.DType().IsInt() {
		exceptions.Panicf("qwen/gomlx: tokens must be integer [B,T]")
	}
	x := graph.Gather(m.node("model.embed_tokens.weight", tokens.Graph()), graph.InsertAxes(tokens, -1))
	for layer := 0; layer < m.value("num_hidden_layers"); layer++ {
		x = m.layer(x, layer, sequenceLengths, nil)
	}
	x = m.rmsNorm(x, "model.norm.weight")
	output := "lm_head.weight"
	if _, exists := m.variables[output]; !exists && m.overrides["lm_head"] == nil {
		output = "model.embed_tokens.weight"
	}
	return m.linear(x, output)
}

// LogitsSegmented builds packed-sequence logits with block-causal attention.
func (m *Model) LogitsSegmented(tokens, segmentIDs *graph.Node) *graph.Node {
	if m == nil || tokens == nil || segmentIDs == nil || tokens.Rank() != 2 || segmentIDs.Rank() != 2 || !tokens.DType().IsInt() || !segmentIDs.DType().IsInt() || !equalDimensions(tokens.Shape().Dimensions, segmentIDs.Shape().Dimensions) {
		exceptions.Panicf("qwen/gomlx: tokens and segment IDs must be matching integer [B,T]")
	}
	x := graph.Gather(m.node("model.embed_tokens.weight", tokens.Graph()), graph.InsertAxes(tokens, -1))
	mask := segmentCausalMask(tokens, segmentIDs)
	for layer := 0; layer < m.value("num_hidden_layers"); layer++ {
		x = m.layer(x, layer, nil, mask)
	}
	x = m.rmsNorm(x, "model.norm.weight")
	output := "lm_head.weight"
	if _, exists := m.variables[output]; !exists && m.overrides["lm_head"] == nil {
		output = "model.embed_tokens.weight"
	}
	return m.linear(x, output)
}

func (m *Model) layer(x *graph.Node, layer int, sequenceLengths, attentionMask *graph.Node) *graph.Node {
	prefix := fmt.Sprintf("model.layers.%d", layer)
	normalized := m.rmsNorm(x, prefix+".input_layernorm.weight")
	heads := m.value("num_attention_heads")
	kvHeads := m.value("num_key_value_heads")
	headDim := m.headDimension()
	q := graph.Reshape(m.linear(normalized, prefix+".self_attn.q_proj.weight"), normalized.Shape().Dim(0), normalized.Shape().Dim(1), heads, headDim)
	k := graph.Reshape(m.linear(normalized, prefix+".self_attn.k_proj.weight"), normalized.Shape().Dim(0), normalized.Shape().Dim(1), kvHeads, headDim)
	v := graph.Reshape(m.linear(normalized, prefix+".self_attn.v_proj.weight"), normalized.Shape().Dim(0), normalized.Shape().Dim(1), kvHeads, headDim)
	positions := graph.Iota(x.Graph(), shapes.Make(dtypes.Int32, x.Shape().Dim(1)), 0)
	rope := pos.NewRoPE(m.floatValue("rope_theta", 10000))
	q, k = rope.EncodeQK(q, k, positions, 1)
	options := attention.CoreOptions{UseCausalMask: true, QuerySeqLen: sequenceLengths, KVSeqLen: sequenceLengths}
	if attentionMask != nil {
		options = attention.CoreOptions{AttentionMask: attentionMask}
	}
	attentionOutput, _ := attention.Core(q, k, v, attention.LayoutBSHD, options)
	attentionOutput = graph.Reshape(attentionOutput, x.Shape().Dim(0), x.Shape().Dim(1), heads*headDim)
	x = graph.Add(x, m.linear(attentionOutput, prefix+".self_attn.o_proj.weight"))
	normalized = m.rmsNorm(x, prefix+".post_attention_layernorm.weight")
	gate := activation.Swish(m.linear(normalized, prefix+".mlp.gate_proj.weight"))
	up := m.linear(normalized, prefix+".mlp.up_proj.weight")
	return graph.Add(x, m.linear(graph.Mul(gate, up), prefix+".mlp.down_proj.weight"))
}

func segmentCausalMask(tokens, segmentIDs *graph.Node) *graph.Node {
	batch, sequence := tokens.Shape().Dim(0), tokens.Shape().Dim(1)
	querySegments := graph.BroadcastToDims(graph.InsertAxes(segmentIDs, -1), batch, sequence, sequence)
	keySegments := graph.BroadcastToDims(graph.InsertAxes(segmentIDs, 1), batch, sequence, sequence)
	sameSegment := graph.Equal(querySegments, keySegments)
	positions := graph.Iota(tokens.Graph(), shapes.Make(dtypes.Int32, sequence), 0)
	queries := graph.BroadcastToDims(graph.Reshape(positions, 1, sequence, 1), batch, sequence, sequence)
	keys := graph.BroadcastToDims(graph.Reshape(positions, 1, 1, sequence), batch, sequence, sequence)
	return graph.InsertAxes(graph.LogicalAnd(sameSegment, graph.GreaterOrEqual(queries, keys)), 2)
}

func (m *Model) linear(input *graph.Node, name string) *graph.Node {
	module := strings.TrimSuffix(name, ".weight")
	if replacement, exists := m.overrides[module]; exists {
		scope, _ := scopeForTensor(m.scope, name)
		return replacement.Apply(scope, input)
	}
	weight := m.node(name, input.Graph())
	return nn.Dense(input, weight, m.bias(name, input.Graph()), compute.DenseLayoutOutputsInput)
}

func (m *Model) bias(weight string, g *graph.Graph) *graph.Node {
	bias := strings.TrimSuffix(weight, ".weight") + ".bias"
	if variable, exists := m.variables[bias]; exists {
		return variable.NodeValue(g)
	}
	return nil
}

func (m *Model) rmsNorm(input *graph.Node, weight string) *graph.Node {
	mean := graph.ReduceAndKeep(graph.Square(input), graph.ReduceMean, -1)
	normalized := graph.Div(input, graph.Sqrt(graph.AddScalar(mean, m.floatValue("rms_norm_eps", 1e-6))))
	scale := graph.Reshape(m.node(weight, input.Graph()), 1, 1, input.Shape().Dim(-1))
	return graph.Mul(normalized, scale)
}

func (m *Model) node(name string, g *graph.Graph) *graph.Node {
	variable, exists := m.variables[name]
	if !exists {
		exceptions.Panicf("qwen/gomlx: missing variable %q", name)
	}
	return variable.NodeValue(g)
}

func (m *Model) value(name string) int { return m.config.Values[name].(int) }

func (m *Model) headDimension() int {
	if value, exists := m.config.Values["head_dim"].(int); exists {
		return value
	}
	return m.value("hidden_size") / m.value("num_attention_heads")
}

func (m *Model) floatValue(name string, fallback float64) float64 {
	value, exists := m.config.Values[name].(float64)
	if !exists {
		return fallback
	}
	return value
}

func scopeForTensor(root *model.Scope, name string) (*model.Scope, string) {
	parts := strings.Split(name, ".")
	scope := root
	for _, part := range parts[:len(parts)-1] {
		scope = scope.At("%s", part)
	}
	return scope, parts[len(parts)-1]
}

func shape(dtype dtypes.DType, dimensions ...int) shapes.Shape {
	return shapes.Make(dtype, dimensions...)
}

func supportedDType(dtype dtypes.DType) bool {
	return dtype == dtypes.Float32 || dtype == dtypes.Float16 || dtype == dtypes.BFloat16
}

func tensorFromFloat32(dtype dtypes.DType, data []float32, dimensions []int) (*tensors.Tensor, error) {
	switch dtype {
	case dtypes.Float32:
		return tensors.FromFlatDataAndDimensions(data, dimensions...), nil
	case dtypes.Float16:
		return tensors.FromFlatDataAndDimensions(float16.FromFloat32s(data...), dimensions...), nil
	case dtypes.BFloat16:
		return tensors.FromFlatDataAndDimensions(bfloat16.FromFloat32s(data...), dimensions...), nil
	default:
		return nil, ErrUnsupportedDType
	}
}

func equalDimensions(left, right []int) bool {
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
