package causallm

import (
	"errors"
	"reflect"
	"testing"
)

func TestRegistryPreservesOrder(t *testing.T) {
	registry, err := NewRegistry([]Module{{Name: "layer.q"}, {Name: "layer.v"}})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(registry.Modules(), []Module{{Name: "layer.q"}, {Name: "layer.v"}}) {
		t.Fatalf("modules = %#v", registry.Modules())
	}
}

func TestRegistryRejectsDuplicates(t *testing.T) {
	_, err := NewRegistry([]Module{{Name: "q"}, {Name: "q"}})
	if !errors.Is(err, ErrDuplicateModule) {
		t.Fatalf("err = %v", err)
	}
}
