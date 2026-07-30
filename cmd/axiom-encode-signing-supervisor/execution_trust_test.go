//go:build darwin || linux

package main

import (
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
)

func TestParseShebangRejectsAmbientInterpreterSelection(t *testing.T) {
	for _, raw := range []string{
		"#!/usr/bin/env python3\n",
		"#!python3 -I\n",
		"#!/opt/axiom/../python -I\n",
		"#!/opt/axiom/python -I extra\n",
		"#!/opt/axiom/python -I\r\n",
	} {
		_, _, script, err := parseShebang([]byte(raw))
		if !script || err == nil {
			t.Fatalf("expected unsafe shebang rejection for %q, got script=%t err=%v", raw, script, err)
		}
	}

	interpreter, argument, script, err := parseShebang(
		[]byte("#!/opt/axiom/runtime/bin/python3.13 -I\n"),
	)
	if err != nil || !script {
		t.Fatalf("expected isolated Python shebang, got script=%t err=%v", script, err)
	}
	if interpreter != "/opt/axiom/runtime/bin/python3.13" || argument != "-I" {
		t.Fatalf("unexpected parsed shebang: %q %q", interpreter, argument)
	}
}

func TestPythonInterpreterNamesAreExact(t *testing.T) {
	for _, name := range []string{"python", "python3", "python3.13", "python3.13t"} {
		if !isPythonInterpreterName(name) {
			t.Fatalf("expected Python interpreter name %q", name)
		}
	}
	for _, name := range []string{"python2", "python-3", "python3.x", "python3.13-debug"} {
		if isPythonInterpreterName(name) {
			t.Fatalf("expected non-Python interpreter rejection for %q", name)
		}
	}
}

func TestNativeExecutableMagicIsNarrow(t *testing.T) {
	for _, header := range [][]byte{
		{0x7f, 'E', 'L', 'F'},
		{0xcf, 0xfa, 0xed, 0xfe},
		{0xca, 0xfe, 0xba, 0xbe},
	} {
		if !isNativeExecutableHeader(header) {
			t.Fatalf("expected native header acceptance for %x", header)
		}
	}
	if isNativeExecutableHeader([]byte("text executable")) {
		t.Fatal("plain text without a shebang must not be accepted as native")
	}
}

func TestDefaultTrustPolicyRejectsCurrentUserTree(t *testing.T) {
	if trustPolicyAllowsWritablePath() {
		t.Skip("fixture build intentionally permits current-user paths")
	}
	if os.Geteuid() == 0 {
		t.Skip("test requires a non-root process")
	}
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "axiom-encode")
	if err := os.WriteFile(path, []byte("#!/usr/bin/python3 -I\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	_, _, err = inspectTrustedExecutable(path)
	if err == nil || !strings.Contains(err.Error(), "root-owned") {
		t.Fatalf("expected current-user ownership rejection, got %v", err)
	}
}

func TestDefaultTrustPolicyAcceptsProtectedSystemNativeExecutable(t *testing.T) {
	if trustPolicyAllowsWritablePath() {
		t.Skip("fixture build exercises its own success path")
	}
	if os.Geteuid() == 0 {
		t.Skip("production policy deliberately rejects a root caller")
	}
	path := "/usr/bin/env"
	info, err := os.Lstat(path)
	if err != nil || info.Mode()&os.ModeSymlink != 0 {
		t.Skip("platform does not expose /usr/bin/env as a regular native executable")
	}
	for current := path; ; current = filepath.Dir(current) {
		if syscall.Access(current, accessWriteMode) == nil {
			t.Skipf(
				"host does not protect %s from writes by the invoking user; "+
					"the production-fixture test covers the protected success path",
				current,
			)
		}
		parent := filepath.Dir(current)
		if parent == current {
			break
		}
	}
	trusted, err := validateTrustedNativeExecutable(path)
	if err != nil {
		t.Fatalf("expected protected system executable acceptance: %v", err)
	}
	if trusted != path {
		t.Fatalf("unexpected trusted path %q", trusted)
	}
}

func TestPathContainmentUsesComponents(t *testing.T) {
	root := filepath.Join(string(os.PathSeparator), "opt", "axiom", "runtime")
	if !pathContains(root, filepath.Join(root, "bin", "python3")) {
		t.Fatal("expected child containment")
	}
	if pathContains(root, root+"-attacker/bin/python3") {
		t.Fatal("path prefix must not count as containment")
	}
}
