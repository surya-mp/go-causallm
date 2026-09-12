# Test Report

Date: 2026-09-12
Host: Windows, PowerShell
GPU: NVIDIA GeForce RTX 3050 Laptop GPU detected by `nvidia-smi`; CUDA version reported by driver: 13.1.

## Commands Run

```powershell
$env:GOCACHE=(Join-Path (Get-Location) '.gocache')
$env:GOMODCACHE=(Join-Path (Get-Location) '.gomodcache')
go test ./...
```

## Result

Pass.

Validated packages:

- `github.com/surya-mp/go-causallm`
- `github.com/surya-mp/go-causallm/architectures/qwen`
- `github.com/surya-mp/go-causallm/architectures/qwen/gomlx`
- `github.com/surya-mp/go-causallm/architectures/qwenmoe`

## CUDA/GPU Notes

This repository does not contain a native CUDA backend or CUDA-tagged tests. GPU-adjacent coverage is through the GoMLX Qwen bridge package, which compiled and passed in this environment. No CUDA kernels were executed by this repo's test suite.
