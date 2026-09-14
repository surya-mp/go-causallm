//go:build gomlx

package gomlx

import (
	"math/rand"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/surya-mp/go-causallm/architectures/qwen"
	peftgomlx "github.com/surya-mp/go-peft/backends/gomlx"
	"github.com/surya-mp/go-peft/format/huggingface"
	"github.com/surya-mp/go-peft/lora"
)

func TestLoadSessionLoadsQwenCheckpoint(t *testing.T) {
	config, err := qwen.ParseConfig([]byte(tinyQwenConfigJSON))
	if err != nil {
		t.Fatal(err)
	}
	session, err := LoadSession(writeCheckpoint(t, config), SessionOptions{DType: dtypes.Float32})
	if err != nil {
		t.Fatal(err)
	}
	if session.Store == nil || session.Model == nil || session.Config.Architecture != "qwen" {
		t.Fatalf("incomplete session: %#v", session)
	}
	if _, ok := session.Model.Variable("model.embed_tokens.weight"); !ok {
		t.Fatal("embedding tensor was not loaded")
	}
}

func TestLoadSessionReportsProgress(t *testing.T) {
	config, err := qwen.ParseConfig([]byte(tinyQwenConfigJSON))
	if err != nil {
		t.Fatal(err)
	}
	var messages []string
	session, err := LoadSession(writeCheckpoint(t, config), SessionOptions{
		DType:    dtypes.Float32,
		Progress: func(message string) { messages = append(messages, message) },
	})
	if err != nil {
		t.Fatal(err)
	}
	if session == nil || len(messages) < 5 {
		t.Fatalf("session=%#v messages=%v", session, messages)
	}
	if messages[0] != "qwen/gomlx: reading config.json" {
		t.Fatalf("first progress message = %q", messages[0])
	}
}

func TestLoadLoRAAdapterInjectsAndLoadsPEFTAdapter(t *testing.T) {
	config, err := qwen.ParseConfig([]byte(tinyQwenConfigJSON))
	if err != nil {
		t.Fatal(err)
	}
	source, err := New(model.NewStore().RootScope().At("qwen"), config, dtypes.Float32)
	if err != nil {
		t.Fatal(err)
	}
	adapter, err := peftgomlx.Inject("adapter", source, lora.Config{
		Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"},
	}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	adapterDir := t.TempDir()
	if err := adapter.Save(adapterDir, huggingface.Metadata{BaseModelNameOrPath: "tiny-qwen", TaskType: "CAUSAL_LM"}); err != nil {
		t.Fatal(err)
	}
	session, err := LoadSession(writeCheckpoint(t, config), SessionOptions{ScopeName: "loaded", DType: dtypes.Float32})
	if err != nil {
		t.Fatal(err)
	}
	loaded, metadata, err := session.LoadLoRAAdapter("adapter", adapterDir, rand.New(rand.NewSource(2)))
	if err != nil {
		t.Fatal(err)
	}
	if metadata.BaseModelNameOrPath != "tiny-qwen" || len(loaded.Layers()) != 1 || len(session.Model.overrides) != 1 {
		t.Fatalf("metadata=%#v layers=%d overrides=%d", metadata, len(loaded.Layers()), len(session.Model.overrides))
	}
}
