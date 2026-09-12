# Development plan

## Purpose

go-causallm defines decoder-only causal-LM architecture plugins and connects
them to GoMLX graph execution and go-peft adapter injection. The core remains
architecture-neutral; each model family is a plugin with its own config,
checkpoint naming map, graph builder, and compatibility tests.

## Public API goals

- Keep architecture Plugin, Config, Module, Registry, and model lifecycle
  contracts backend-neutral where possible.
- Keep core contracts free of GoMLX and go-peft imports. Put concrete graph and
  adapter wiring in an explicit gomlx subpackage or build-tagged package.
- Define a loader contract that consumes local SafeTensors streams rather than
  downloading files.
- Expose stable named linear modules suitable for go-peft injection.
- Keep adapter and base-weight ownership explicit.

## Development milestones

### M1: core contracts and fixture harness

- Finish architecture-plugin, checkpoint-name-map, graph-builder, and optional
  KV-cache contracts.
- Define a tiny synthetic checkpoint fixture format.
- Add a numerical comparison harness for logits, loss, gradients, and generated
  tokens with backend-specific tolerances.
- Define clear error types for unknown architecture, missing tensor, shape
  mismatch, unsupported dtype, and incompatible adapter target.

### M2: first GoMLX architecture plugin

- Completed: Qwen plugin configuration validation and canonical adapter-name
  mapping plus expected SafeTensors names/shapes, without placing Qwen names
  in the core package.
- Completed: one-tensor-at-a-time SafeTensors loading into a backend sink with
  required-name and shape validation.
- Completed: dense Qwen GoMLX variable sink and causal logits graph using GQA,
  RoPE, RMSNorm, and SwiGLU.
- Use GoMLX generation, KV cache, training, and optimizers directly; do not
  add local wrappers. The next graph work is an architecture-specific
  incremental Qwen callback only when direct GoMLX composition requires it.
- Map config fields to GoMLX transformer, attention, RMSNorm, RoPE, SwiGLU,
  grouped-query attention, causal mask, and tied embeddings.
- Load F32, F16, and BF16 SafeTensors streams into named GoMLX variables.
- Validate every expected tensor name and shape before graph execution.
- Expose module names required by go-peft LoRA and QLoRA injection.

### M3: adapter training and inference

- Completed: go-peft GoMLX LoRA host registry and graph-time replacement for
  dense Qwen projections.
- Completed: native GoMLX NF4 QLoRA, including optional double quantization,
  replacing every frozen dense projection; only selected targets retain LoRA
  A/B, and full-precision projection variables are released.
- Completed: native GoMLX trainer smoke test verifies a Qwen LoRA optimizer
  step updates adapter variables while base weights remain frozen.
- Completed: packed Qwen forward graph accepts segment IDs and builds its own
  block-causal attention mask.
- Ensure GoMLX optimizers see A/B and configured bias only.
- Build masked causal cross-entropy graphs used by go-sft.
- Keep GoMLX generation and KV-cache API composition outside this package.

### M4: additional architecture plugins

- Completed: Qwen3-MoE config parsing, sparse-layer selection, and strict
  per-expert SafeTensors inventory validation.
- Add a new plugin only with a real checkpoint fixture and golden comparison.
- Keep all architecture-specific loader names, configs, and graph choices in
  that plugin directory.
- Do not generalize speculative fields into the core API.

## Test plan

- Unit-test registry ordering, duplicate rejection, config parsing, tensor-name
  mapping, and shape validation.
- Compare tiny random-model logits and gradients against Python fixtures.
- Test a synthetic SafeTensors load without external downloads.
- Test LoRA A/B parameter discovery, forward update, save/load, and frozen base
  variables through current go-peft integration.
- Test QLoRA NF4/double quantization, frozen base replacement, and A/B-only
  trainables separately from F32 LoRA.
- Run GoMLX-tagged tests on supported CPU/GPU backends and race-test only the
  portions GoMLX supports.
- Add a gated GPU smoke test that loads a public small model revision.

## Integration plan

- Consume a go-hfhub snapshot and go-peft local model stream.
- Accept token IDs and masks from go-tokenizer and go-sft.
- Provide a GoMLX model registry consumed by go-peft backends/gomlx.
- Provide logits and KV-cache entry points consumed by the GoMLX generator.
- Keep applications responsible for choosing a model plugin and run policy.

## Performance plan

- Measure load time, peak host memory, peak device memory, prefill tokens per
  second, decode tokens per second, and training tokens per second.
- Use GoMLX fused attention and graph bucketing before writing custom CUDA code.
- Add custom kernels only after a measured, isolated bottleneck.
- Never silently train on CPU when a GPU backend was requested.

## Release criteria

- One architecture plugin has complete config, load, inference, LoRA, QLoRA,
  training-graph, and generation compatibility tests.
- PEFT adapters round-trip with the current go-peft package.
- Model, adapter, and tokenizer revisions are recorded by the application
  checkpoint manifest.
- Supported architectures and numerical tolerances are documented.

## Non-goals

- Hugging Face transport, tokenization, SFT batch policy, generic optimizers,
  or HTTP serving.
