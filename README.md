# go-causallm

Decoder-only causal-LM architecture and module-registry contracts.

## Install

```sh
go get github.com/surya-mp/go-causallm
```

See the [API reference](docs/api.md).

`GreedyGenerate` provides backend-neutral autoregressive decoding over a
caller-supplied next-token function. It handles max-new-token, EOS, PAD, custom
stop-token, and context-cancellation policy.

Architecture plugins map a Hugging Face checkpoint to GoMLX graphs and expose
stable linear names to go-peft. The core is model-family neutral. The initial
Qwen plugin validates configuration, exposes canonical adapter names, and
builds an expected Hugging Face SafeTensors inventory. `LoadSafeTensors`
streams and validates a checkpoint into a backend tensor sink.

`architectures/qwen/gomlx` provides the first dense Qwen2/Qwen3 bridge: frozen
named variables, F32/F16/BF16 loading, causal forward logits, and
`LoadSession` for Hugging Face-style Qwen checkpoint directories. A session
loads `config.json`, creates the GoMLX store and Qwen graph, streams
SafeTensors into model variables, and can attach a standard Hugging Face PEFT
LoRA adapter with `Session.LoadLoRAAdapter`. Qwen3-MoE has an exact
configuration and SafeTensors inventory plugin in
`architectures/qwenmoe`; its sparse dispatch graph remains separate.

Set `qwen.LoadOptions.Progress` or `gomlx.SessionOptions.Progress` to display
checkpoint-load progress. Messages are human-readable diagnostics, not a
machine-stable event protocol.

With `-tags gomlx`, the dense bridge is also a native go-peft LoRA host for
Qwen's q/k/v/o and gate/up/down projections, plus an untied `lm_head`. QLoRA
quantizes every frozen projection to NF4 (optionally double-quantized), while
only configured targets receive LoRA A/B; full-precision projection variables
are released. `Model.KVCache`, `Model.LogitsWithKVCache`, and
`Model.Generator` wire dense Qwen to GoMLX's KV-cache generator. Cached
generation requires a GoMLX backend that implements the cache update ops, such
as `DynamicUpdateSlice`.

`LogitsSegmented` accepts packed sequence segment IDs and builds the required
block-causal attention mask directly in the Qwen graph.

Native GoMLX NF4 QLoRA requires even projection output widths because its
packed `Uint4` runtime has no safe odd-width slice operation.

## Scope

Dense Qwen2/Qwen3 is the runnable GoMLX architecture in this release.
Qwen3-MoE validates configuration and checkpoint inventories, but sparse MoE
forward/training dispatch is not implemented. Loading, tokenization, SFT
batching, optimization, generation, and serving stay in their dedicated
libraries or application.
