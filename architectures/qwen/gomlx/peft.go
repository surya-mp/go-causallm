//go:build gomlx

package gomlx

import (
	"errors"
	"fmt"
	"strings"

	"github.com/surya-mp/go-causallm/architectures/qwen"
	peftgomlx "github.com/surya-mp/go-peft/backends/gomlx"
)

// LinearModules implements go-peft's native GoMLX LoRA host contract.
func (m *Model) LinearModules() ([]peftgomlx.Module, error) {
	if m == nil {
		return nil, ErrInvalidModel
	}
	specs, err := qwen.TensorSpecs(m.config)
	if err != nil {
		return nil, err
	}
	modules := make([]peftgomlx.Module, 0, len(specs))
	for _, spec := range specs {
		if !linearWeight(spec.Name) {
			continue
		}
		weight := m.variables[spec.Name]
		if weight == nil {
			return nil, fmt.Errorf("%w: missing variable %s", ErrInvalidModel, spec.Name)
		}
		scope, _ := scopeForTensor(m.scope, spec.Name)
		name := strings.TrimSuffix(spec.Name, ".weight")
		var biasName = strings.TrimSuffix(spec.Name, ".weight") + ".bias"
		modules = append(modules, peftgomlx.Module{Name: name, Scope: scope, Weight: weight, Bias: m.variables[biasName]})
	}
	return modules, nil
}

// ReplaceLoRALinearModules implements go-peft's native GoMLX LoRA host contract.
func (m *Model) ReplaceLoRALinearModules(replacements []peftgomlx.Replacement) error {
	if m == nil {
		return ErrInvalidModel
	}
	next := make(map[string]LinearOverride, len(replacements))
	for _, replacement := range replacements {
		if replacement.Name == "" || replacement.Layer == nil || !linearWeight(replacement.Name+".weight") {
			return errors.New("qwen/gomlx: invalid LoRA replacement")
		}
		if _, exists := next[replacement.Name]; exists {
			return fmt.Errorf("qwen/gomlx: duplicate LoRA replacement %q", replacement.Name)
		}
		next[replacement.Name] = replacement.Layer
	}
	for name, replacement := range next {
		m.overrides[name] = replacement
	}
	return nil
}

func linearWeight(name string) bool {
	if name == "lm_head.weight" {
		return true
	}
	for _, suffix := range []string{
		".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight", ".self_attn.o_proj.weight",
		".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight",
	} {
		if strings.HasSuffix(name, suffix) {
			return true
		}
	}
	return false
}
