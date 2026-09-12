//go:build gomlx

package gomlx

import (
	"errors"
	"fmt"
	"math/rand"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/nn"
	"github.com/surya-mp/go-causallm/architectures/qwen"
	peftgomlx "github.com/surya-mp/go-peft/backends/gomlx"
	"github.com/surya-mp/go-peft/lora"
	"github.com/surya-mp/go-peft/qlora"
)

var (
	// ErrQLoRANotPrepared reports NF4 injection without prepared base weights.
	ErrQLoRANotPrepared = errors.New("qwen/gomlx: QLoRA weights are not prepared")
	// ErrQLoRAConflict reports a projection already replaced by another adapter.
	ErrQLoRAConflict = errors.New("qwen/gomlx: projection already replaced")
)

// InjectQLoRA quantizes every frozen dense Qwen projection to NF4 and injects
// LoRA only into the configured target modules. Replaced base variables are released.
func InjectQLoRA(name string, decoder *Model, config qlora.Config, rng *rand.Rand) (*peftgomlx.NF4Adapter, error) {
	if decoder == nil || rng == nil {
		return nil, ErrInvalidModel
	}
	if err := prepareNF4(decoder, config); err != nil {
		return nil, err
	}
	if err := installFrozenNF4(decoder, config.LoRA.TargetModules); err != nil {
		return nil, err
	}
	return peftgomlx.InjectNF4(name, decoder, config, rng)
}

// NF4LinearModules implements go-peft's native GoMLX QLoRA host contract.
func (m *Model) NF4LinearModules() ([]peftgomlx.NF4Module, error) {
	if m == nil {
		return nil, ErrInvalidModel
	}
	specs, err := qwen.TensorSpecs(m.config)
	if err != nil {
		return nil, err
	}
	modules := make([]peftgomlx.NF4Module, 0, len(m.quantized))
	for _, spec := range specs {
		if !linearWeight(spec.Name) {
			continue
		}
		name := strings.TrimSuffix(spec.Name, ".weight")
		stored, exists := m.quantized[name]
		if !exists {
			continue
		}
		weight, ok := stored.(*peftgomlx.NF4Weight)
		if !ok {
			return nil, ErrQLoRANotPrepared
		}
		scope, _ := scopeForTensor(m.scope, spec.Name)
		bias := m.variables[strings.TrimSuffix(spec.Name, ".weight")+".bias"]
		modules = append(modules, peftgomlx.NF4Module{Name: name, Scope: scope, Weight: weight, Bias: bias})
	}
	return modules, nil
}

// ReplaceNF4LinearModules implements go-peft's native GoMLX QLoRA host contract.
func (m *Model) ReplaceNF4LinearModules(replacements []peftgomlx.NF4Replacement) error {
	if m == nil {
		return ErrInvalidModel
	}
	next := make(map[string]LinearOverride, len(replacements))
	for _, replacement := range replacements {
		if replacement.Name == "" || replacement.Layer == nil || !linearWeight(replacement.Name+".weight") {
			return errors.New("qwen/gomlx: invalid NF4 replacement")
		}
		if _, exists := m.quantized[replacement.Name]; !exists {
			return fmt.Errorf("%w: %s", ErrQLoRANotPrepared, replacement.Name)
		}
		if _, exists := m.overrides[replacement.Name]; exists {
			return fmt.Errorf("%w: %s", ErrQLoRAConflict, replacement.Name)
		}
		if _, exists := next[replacement.Name]; exists {
			return fmt.Errorf("qwen/gomlx: duplicate NF4 replacement %q", replacement.Name)
		}
		next[replacement.Name] = replacement.Layer
	}
	for name, replacement := range next {
		weightName := name + ".weight"
		variable := m.variables[weightName]
		if variable == nil {
			return fmt.Errorf("%w: %s", ErrUnknownTensor, weightName)
		}
		if err := m.scope.Store().DeleteVariable(variable.Path()); err != nil {
			return err
		}
		delete(m.variables, weightName)
		delete(m.quantized, name)
		m.overrides[name] = replacement
	}
	return nil
}

func prepareNF4(m *Model, config qlora.Config) error {
	if err := config.Validate(); err != nil {
		return err
	}
	if config.Quantization != qlora.QuantizationNF4 {
		return errors.New("qwen/gomlx: native QLoRA requires NF4")
	}
	prepared := make(map[string]any)
	modules, err := m.LinearModules()
	if err != nil {
		return err
	}
	matched := 0
	for _, module := range modules {
		if lora.Matches(module.Name, config.LoRA.TargetModules) {
			matched++
		}
	}
	if matched == 0 {
		return peftgomlx.ErrNoTargetModules
	}
	for _, module := range modules {
		if _, exists := m.overrides[module.Name]; exists {
			return fmt.Errorf("%w: %s", ErrQLoRAConflict, module.Name)
		}
		values, err := outputInputValues(module.Weight)
		if err != nil {
			return fmt.Errorf("qwen/gomlx: read %s: %w", module.Name, err)
		}
		out, in := module.Weight.Shape().Dimensions[0], module.Weight.Shape().Dimensions[1]
		if out&1 != 0 {
			return errors.New("qwen/gomlx: NF4 projection output features must be even")
		}
		var weight *peftgomlx.NF4Weight
		if config.DoubleQuant {
			weight, err = peftgomlx.QuantizeNF4WeightDouble(in, out, config.BlockSize, config.ScaleBlockSize, values)
		} else {
			weight, err = peftgomlx.QuantizeNF4Weight(in, out, config.BlockSize, values)
		}
		if err != nil {
			return err
		}
		prepared[module.Name] = weight
	}
	for name, weight := range prepared {
		m.quantized[name] = weight
	}
	return nil
}

func installFrozenNF4(m *Model, targets []string) error {
	specs, err := qwen.TensorSpecs(m.config)
	if err != nil {
		return err
	}
	type pending struct {
		name    string
		weight  string
		replace LinearOverride
	}
	next := make([]pending, 0, len(m.quantized))
	for _, spec := range specs {
		if !linearWeight(spec.Name) {
			continue
		}
		name := strings.TrimSuffix(spec.Name, ".weight")
		if lora.Matches(name, targets) {
			continue
		}
		stored, exists := m.quantized[name]
		if !exists {
			return fmt.Errorf("%w: %s", ErrQLoRANotPrepared, name)
		}
		weight, ok := stored.(*peftgomlx.NF4Weight)
		if !ok {
			return fmt.Errorf("%w: %s", ErrQLoRANotPrepared, name)
		}
		variable := m.variables[spec.Name]
		if variable == nil {
			return fmt.Errorf("%w: %s", ErrUnknownTensor, spec.Name)
		}
		scope, _ := scopeForTensor(m.scope, spec.Name)
		base, err := newNF4BaseLinear(scope, weight, m.variables[name+".bias"])
		if err != nil {
			return fmt.Errorf("qwen/gomlx: replace %s: %w", name, err)
		}
		next = append(next, pending{name: name, weight: spec.Name, replace: base})
	}
	for _, replacement := range next {
		variable := m.variables[replacement.weight]
		if err := m.scope.Store().DeleteVariable(variable.Path()); err != nil {
			return err
		}
		delete(m.variables, replacement.weight)
		delete(m.quantized, replacement.name)
		m.overrides[replacement.name] = replacement.replace
	}
	return nil
}

// nf4BaseLinear owns a frozen NF4 projection without LoRA variables.
type nf4BaseLinear struct {
	packed, scales, scaleCodes, scaleScales, bias *model.Variable
	in, out, block                                int
}

func newNF4BaseLinear(scope *model.Scope, weight *peftgomlx.NF4Weight, bias *model.Variable) (*nf4BaseLinear, error) {
	if scope == nil || weight == nil || weight.InputFeatures <= 0 || weight.OutputFeatures <= 0 || weight.BlockSize <= 0 {
		return nil, errors.New("invalid NF4 base projection")
	}
	if bias != nil && (bias.Shape().Rank() != 1 || bias.Shape().Dimensions[0] != weight.OutputFeatures) {
		return nil, errors.New("NF4 bias shape must be [out_features]")
	}
	blocks := (weight.OutputFeatures + weight.BlockSize - 1) / weight.BlockSize
	qScope := scope.In("nf4_base")
	base := &nf4BaseLinear{
		packed: qScope.VariableWithValue("packed", tensors.FromFlatDataAndDimensions(weight.Packed, weight.InputFeatures, (weight.OutputFeatures+1)/2)).SetTrainable(false),
		bias:   bias, in: weight.InputFeatures, out: weight.OutputFeatures, block: weight.BlockSize,
	}
	if len(weight.ScaleCodes) == 0 {
		base.scales = qScope.VariableWithValue("scales", tensors.FromFlatDataAndDimensions(weight.Scales, weight.InputFeatures, blocks)).SetTrainable(false)
	} else {
		groups := (blocks + weight.ScaleBlockSize - 1) / weight.ScaleBlockSize
		base.scaleCodes = qScope.VariableWithValue("scale_codes", tensors.FromFlatDataAndDimensions(weight.ScaleCodes, weight.InputFeatures, blocks)).SetTrainable(false)
		base.scaleScales = qScope.VariableWithValue("scale_scales", tensors.FromFlatDataAndDimensions(weight.ScaleScales, weight.InputFeatures, groups)).SetTrainable(false)
	}
	if bias != nil {
		bias.SetTrainable(false)
	}
	return base, nil
}

func (l *nf4BaseLinear) Apply(_ *model.Scope, input *graph.Node) *graph.Node {
	packed := graph.Bitcast(l.packed.NodeValue(input), dtypes.Uint4)
	weights := graph.Reshape(packed, l.in, l.out)
	return nn.QuantizedDense(input, weights, &graph.Quantization{
		Scheme: compute.QuantNF4, Scale: l.scalesFor(input), BlockAxis: 1, BlockSize: l.block,
	}, nf4NodeValue(l.bias, input))
}

func (l *nf4BaseLinear) scalesFor(input *graph.Node) *graph.Node {
	if l.scales != nil {
		return l.scales.NodeValue(input)
	}
	blocks := (l.out + l.block - 1) / l.block
	groups := l.scaleScales.Shape().Dimensions[1]
	groupSize := (blocks + groups - 1) / groups
	indices := make([]int32, blocks)
	for index := range indices {
		indices[index] = int32(index / groupSize)
	}
	group := graph.Reshape(graph.Const(input.Graph(), indices), blocks, 1)
	expanded := graph.Gather(graph.Transpose(l.scaleScales.NodeValue(input), 0, 1), group)
	expanded = graph.Transpose(expanded, 0, 1)
	return graph.Mul(graph.ConvertDType(l.scaleCodes.NodeValue(input), dtypes.Float32), expanded)
}

func nf4NodeValue(variable *model.Variable, input *graph.Node) *graph.Node {
	if variable == nil {
		return nil
	}
	return variable.NodeValue(input)
}

// outputInputValues returns a transposed [input, output] F32 view for native NF4 quantization.
func outputInputValues(variable *model.Variable) ([]float32, error) {
	value, err := variable.Value()
	if err != nil {
		return nil, err
	}
	var source []float32
	switch variable.DType() {
	case dtypes.Float32:
		source, err = tensors.CopyFlatData[float32](value)
	case dtypes.Float16:
		var half []float16.Float16
		half, err = tensors.CopyFlatData[float16.Float16](value)
		source = make([]float32, len(half))
		for index, number := range half {
			source[index] = number.Float32()
		}
	case dtypes.BFloat16:
		var half []bfloat16.BFloat16
		half, err = tensors.CopyFlatData[bfloat16.BFloat16](value)
		source = make([]float32, len(half))
		for index, number := range half {
			source[index] = number.Float32()
		}
	default:
		return nil, ErrUnsupportedDType
	}
	if err != nil {
		return nil, err
	}
	dimensions := variable.Shape().Dimensions
	if len(dimensions) != 2 {
		return nil, qwen.ErrTensorShape
	}
	out, in := dimensions[0], dimensions[1]
	transposed := make([]float32, len(source))
	for output := 0; output < out; output++ {
		for input := 0; input < in; input++ {
			transposed[input*out+output] = source[output*in+input]
		}
	}
	return transposed, nil
}
