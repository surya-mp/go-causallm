//go:build gomlx

package gomlx

import (
	"fmt"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/layers/attention"
	"github.com/gomlx/gomlx/ml/layers/attention/kvcache"
	"github.com/gomlx/gomlx/ml/layers/attention/pos"
	"github.com/gomlx/gomlx/ml/model"
	transformergenerate "github.com/gomlx/gomlx/ml/zoo/transformer/generate"
)

// KVCache returns a GoMLX KV-cache configuration for this Qwen decoder.
func (m *Model) KVCache() *kvcache.KVCache {
	scopes := make([]string, 0, m.value("num_hidden_layers"))
	layerTypes := make(map[string]attention.LayerType, m.value("num_hidden_layers"))
	for layer := 0; layer < m.value("num_hidden_layers"); layer++ {
		scope := m.attentionScope(layer).Scope()
		scopes = append(scopes, scope)
		layerTypes[scope] = attention.GlobalLayer
	}
	return kvcache.NewKVCache().
		WithOrderedScopes(scopes).
		WithLayerTypes(layerTypes).
		WithGlobalHeadDim(m.headDimension())
}

// Generator returns GoMLX's transformer generator wired to Qwen's incremental
// KV-cache forward path.
func (m *Model) Generator() *transformergenerate.Generator {
	return transformergenerate.New(transformergenerate.KVCacheModelFn(m.LogitsWithKVCache)).
		WithKVCache(m.KVCache(), m.value("num_key_value_heads"), m.headDimension(), m.dtype)
}

// LogitsWithKVCache builds incremental Qwen logits and returns updated cache nodes.
func (m *Model) LogitsWithKVCache(scope *model.Scope, tokens, position *graph.Node, cache kvcache.KVCacheNodes) (*graph.Node, kvcache.KVCacheNodes) {
	if m == nil || tokens == nil || position == nil || cache == nil || tokens.Rank() != 2 || !tokens.DType().IsInt() {
		panic("qwen/gomlx: cached tokens must be integer [B,T]")
	}
	x := graph.Gather(m.node("model.embed_tokens.weight", tokens.Graph()), graph.InsertAxes(tokens, -1))
	for layer := 0; layer < m.value("num_hidden_layers"); layer++ {
		x = m.layerWithKVCache(x, layer, position, cache)
	}
	x = m.rmsNorm(x, "model.norm.weight")
	output := "lm_head.weight"
	if _, exists := m.variables[output]; !exists && m.overrides["lm_head"] == nil {
		output = "model.embed_tokens.weight"
	}
	return m.linear(x, output), cache
}

func (m *Model) layerWithKVCache(x *graph.Node, layer int, position *graph.Node, cache kvcache.KVCacheNodes) *graph.Node {
	prefix := fmt.Sprintf("model.layers.%d", layer)
	normalized := m.rmsNorm(x, prefix+".input_layernorm.weight")
	heads := m.value("num_attention_heads")
	kvHeads := m.value("num_key_value_heads")
	headDim := m.headDimension()
	q := graph.Reshape(m.linear(normalized, prefix+".self_attn.q_proj.weight"), normalized.Shape().Dim(0), normalized.Shape().Dim(1), heads, headDim)
	k := graph.Reshape(m.linear(normalized, prefix+".self_attn.k_proj.weight"), normalized.Shape().Dim(0), normalized.Shape().Dim(1), kvHeads, headDim)
	v := graph.Reshape(m.linear(normalized, prefix+".self_attn.v_proj.weight"), normalized.Shape().Dim(0), normalized.Shape().Dim(1), kvHeads, headDim)
	positions := graph.Add(graph.Iota(x.Graph(), shapes.Make(dtypes.Int32, x.Shape().Dim(1)), 0), position)
	rope := pos.NewRoPE(m.floatValue("rope_theta", 10000))
	q, k = rope.EncodeQK(q, k, positions, 1)
	attentionScope := m.attentionScope(layer)
	m.KVCache().Update(attentionScope, cache, k, v, position)
	cachedK, cachedV := m.KVCache().Get(cache, attentionScope.Scope())
	cachedK = graph.TransposeAllDims(cachedK, 0, 2, 1, 3)
	cachedV = graph.TransposeAllDims(cachedV, 0, 2, 1, 3)
	mask := graph.InsertAxes(m.KVCache().BuildAttentionMask(attentionScope, cache, q, position, true, 0), 2)
	attentionOutput, _ := attention.Core(q, cachedK, cachedV, attention.LayoutBSHD, attention.CoreOptions{AttentionMask: mask})
	attentionOutput = graph.Reshape(attentionOutput, x.Shape().Dim(0), x.Shape().Dim(1), heads*headDim)
	x = graph.Add(x, m.linear(attentionOutput, prefix+".self_attn.o_proj.weight"))
	normalized = m.rmsNorm(x, prefix+".post_attention_layernorm.weight")
	gate := activation.Swish(m.linear(normalized, prefix+".mlp.gate_proj.weight"))
	up := m.linear(normalized, prefix+".mlp.up_proj.weight")
	return graph.Add(x, m.linear(graph.Mul(gate, up), prefix+".mlp.down_proj.weight"))
}

func (m *Model) attentionScope(layer int) *model.Scope {
	scope, _ := scopeForTensor(m.scope, fmt.Sprintf("model.layers.%d.self_attn.cache", layer))
	return scope
}
