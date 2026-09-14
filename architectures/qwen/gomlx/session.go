//go:build gomlx

package gomlx

import (
	"errors"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/surya-mp/go-causallm"
	"github.com/surya-mp/go-causallm/architectures/qwen"
	peftgomlx "github.com/surya-mp/go-peft/backends/gomlx"
	"github.com/surya-mp/go-peft/format/huggingface"
)

// Session owns a loaded Qwen GoMLX model and its backing variable store.
type Session struct {
	ModelDir string
	Config   causallm.Config
	Store    *model.Store
	Model    *Model
}

// SessionOptions configures LoadSession.
type SessionOptions struct {
	ScopeName string
	DType     dtypes.DType
	Progress  func(string)
}

// LoadSession loads config.json and dense SafeTensors weights from a Hugging
// Face Qwen checkpoint directory into a GoMLX-backed model.
func LoadSession(modelDir string, options SessionOptions) (*Session, error) {
	if modelDir == "" {
		return nil, errors.New("qwen/gomlx: model directory is required")
	}
	if options.ScopeName == "" {
		options.ScopeName = "qwen"
	}
	if !supportedDType(options.DType) {
		return nil, ErrUnsupportedDType
	}
	progressf(options.Progress, "qwen/gomlx: reading config.json")
	data, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return nil, err
	}
	progressf(options.Progress, "qwen/gomlx: parsing config")
	config, err := qwen.ParseConfig(data)
	if err != nil {
		return nil, err
	}
	progressf(options.Progress, "qwen/gomlx: creating variable store")
	store := model.NewStore()
	progressf(options.Progress, "qwen/gomlx: creating graph variables scope=%s dtype=%s", options.ScopeName, options.DType)
	decoder, err := New(store.RootScope().At("%s", options.ScopeName), config, options.DType)
	if err != nil {
		return nil, err
	}
	progressf(options.Progress, "qwen/gomlx: loading SafeTensors")
	if err := qwen.LoadSafeTensorsWithOptions(modelDir, config, decoder, qwen.LoadOptions{Progress: options.Progress}); err != nil {
		return nil, err
	}
	progressf(options.Progress, "qwen/gomlx: session ready")
	return &Session{ModelDir: modelDir, Config: config, Store: store, Model: decoder}, nil
}

func progressf(progress func(string), format string, args ...any) {
	if progress != nil {
		progress(fmt.Sprintf(format, args...))
	}
}

// LoadLoRAAdapter injects a Hugging Face PEFT LoRA adapter into the session's
// Qwen graph, validates the adapter tensors, and copies them into GoMLX variables.
func (s *Session) LoadLoRAAdapter(name, adapterDir string, rng *rand.Rand) (*peftgomlx.Adapter, huggingface.Metadata, error) {
	if s == nil || s.Model == nil {
		return nil, huggingface.Metadata{}, ErrInvalidModel
	}
	config, _, err := huggingface.ReadConfig(adapterDir)
	if err != nil {
		return nil, huggingface.Metadata{}, err
	}
	adapter, err := peftgomlx.Inject(name, s.Model, config, rng)
	if err != nil {
		return nil, huggingface.Metadata{}, err
	}
	metadata, err := adapter.Load(adapterDir)
	if err != nil {
		return nil, huggingface.Metadata{}, err
	}
	return adapter, metadata, nil
}
