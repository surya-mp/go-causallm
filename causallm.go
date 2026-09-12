// Package causallm defines architecture-neutral causal language model contracts.
package causallm

import (
	"errors"
	"fmt"
)

var (
	// ErrInvalidConfig reports an invalid architecture configuration.
	ErrInvalidConfig = errors.New("causallm: invalid configuration")
	// ErrDuplicateModule reports duplicate registered module names.
	ErrDuplicateModule = errors.New("causallm: duplicate module")
)

// Config contains architecture-owned configuration values.
type Config struct {
	// Architecture identifies the plugin that owns Values.
	Architecture string
	// Values contains architecture-specific scalar and structured configuration.
	Values map[string]any
}

// Plugin maps one decoder-only architecture to a graph backend.
type Plugin interface {
	// Name identifies the architecture plugin.
	Name() string
	// Validate rejects incompatible architecture settings.
	Validate(Config) error
	// TargetModules returns adapter-injectable linear module suffixes.
	TargetModules(Config) ([]string, error)
}

// Module is one named, replaceable host linear layer.
type Module struct {
	// Name is the stable checkpoint and adapter module name.
	Name string
	// Value is owned by the selected graph backend.
	Value any
}

// Registry validates and exposes host linear modules for adapter injection.
type Registry struct {
	index map[string]Module
	order []string
}

// NewRegistry creates a registry from unique module names.
func NewRegistry(modules []Module) (*Registry, error) {
	registry := &Registry{index: make(map[string]Module, len(modules)), order: make([]string, 0, len(modules))}
	for _, module := range modules {
		if module.Name == "" {
			return nil, ErrInvalidConfig
		}
		if _, exists := registry.index[module.Name]; exists {
			return nil, fmt.Errorf("%w: %s", ErrDuplicateModule, module.Name)
		}
		registry.index[module.Name] = module
		registry.order = append(registry.order, module.Name)
	}
	return registry, nil
}

// Modules returns registry modules in construction order.
func (r *Registry) Modules() []Module {
	result := make([]Module, 0, len(r.order))
	for _, name := range r.order {
		result = append(result, r.index[name])
	}
	return result
}
