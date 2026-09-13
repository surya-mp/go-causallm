# API reference

Canonical API documentation is generated from Go doc comments on
[pkg.go.dev](https://pkg.go.dev/github.com/surya-mp/go-causallm).

| Package / API | Use |
| --- | --- |
| `causallm.NewRegistry` | Register architecture plugins without duplicates. |
| `causallm.GreedyGenerate` | Run backend-neutral token-by-token decoding. |
| `architectures/qwen.ParseConfig` | Validate dense Qwen2/Qwen3 configuration. |
| `architectures/qwen.LoadSafeTensors` | Stream a Qwen checkpoint into a tensor sink. |
| `architectures/qwen/gomlx.New` | Build the dense Qwen GoMLX graph. |
| `gomlx.LoadSession` | Load a dense Qwen checkpoint and its GoMLX variable store. |
| `Session.LoadLoRAAdapter` | Inject and load a Hugging Face PEFT LoRA adapter. |
| `Model.Logits` / `LogitsSegmented` | Build normal or packed causal forward logits. |
| `Model.KVCache` / `Model.Generator` | Build GoMLX cached generation. |
| `Model.LinearModules` | Expose named Qwen projections for go-peft LoRA. |
| `InjectQLoRA` | Install NF4 QLoRA into a dense Qwen GoMLX model. |
| `architectures/qwenmoe` | Validate Qwen3-MoE config and checkpoint inventory. |

Only dense Qwen2/Qwen3 has a runnable forward/training graph. Qwen3-MoE
sparse dispatch is intentionally not exposed as a runnable API yet.
