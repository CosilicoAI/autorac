//go:build (darwin || linux) && signing_supervisor_test_fixture

package main

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

type fixturePythonChain struct {
	runtimeRoot string
	importRoot  string
	packageRoot string
	interpreter string
	bootstrap   string
	executable  string
}

func newFixturePythonChain(t *testing.T) fixturePythonChain {
	t.Helper()
	temporary, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	runtimeRoot := filepath.Join(temporary, "runtime")
	interpreterDirectory := filepath.Join(runtimeRoot, "bin")
	packageRoot := filepath.Join(runtimeRoot, "lib", "python3.13", "site-packages", "axiom_encode")
	if err := os.MkdirAll(interpreterDirectory, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(packageRoot, 0o700); err != nil {
		t.Fatal(err)
	}
	interpreter := filepath.Join(interpreterDirectory, "python3.13")
	if err := os.WriteFile(interpreter, []byte{0x7f, 'E', 'L', 'F'}, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(packageRoot, "__init__.py"), []byte("\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	bootstrap := filepath.Join(packageRoot, "_trusted_signing_bootstrap.py")
	if err := os.WriteFile(bootstrap, []byte("\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	executable := filepath.Join(runtimeRoot, "axiom-encode")
	launcher := []byte("#!" + interpreter + " -I\n")
	if err := os.WriteFile(executable, launcher, 0o700); err != nil {
		t.Fatal(err)
	}
	return fixturePythonChain{
		runtimeRoot: runtimeRoot,
		importRoot:  filepath.Dir(packageRoot),
		packageRoot: packageRoot,
		interpreter: interpreter,
		bootstrap:   bootstrap,
		executable:  executable,
	}
}

func TestFixturePolicyValidatesHermeticPythonChain(t *testing.T) {
	fixture := newFixturePythonChain(t)

	chain, err := validateTrustedExecutionChain(
		fixture.executable,
		[]string{fixture.runtimeRoot},
		[]string{fixture.importRoot},
		fixture.packageRoot,
	)
	if err != nil {
		t.Fatal(err)
	}
	if chain.Executable != fixture.interpreter || chain.PythonInterpreter != fixture.interpreter {
		t.Fatalf("unexpected trusted chain: %#v", chain)
	}
	if chain.PythonPackageRoot != fixture.packageRoot || len(chain.PythonRuntimeRoots) != 1 {
		t.Fatalf("unexpected Python trust roots: %#v", chain)
	}
}

func TestFixtureSupervisorPreservesCompleteSourceUnitArgumentThroughBootstrap(t *testing.T) {
	fixture := newFixturePythonChain(t)
	baseCommand := []string{
		fixture.executable,
		"encode",
		"uk/statute/toy",
		"--apply",
	}
	tests := []struct {
		name    string
		command []string
		count   int
	}{
		{
			name:    "default off leaves encoder argv unchanged",
			command: append([]string(nil), baseCommand...),
			count:   0,
		},
		{
			name: "enabled preserves exactly one appended literal",
			command: append(
				append([]string(nil), baseCommand...),
				"--require-complete-source-unit",
			),
			count: 1,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			arguments := []string{
				"--trusted-signing-roots", filepath.Join(fixture.runtimeRoot, "trust.json"),
				"--trusted-python-runtime-root", fixture.runtimeRoot,
				"--trusted-python-import-root", fixture.importRoot,
				"--trusted-python-package-root", fixture.packageRoot,
				"--",
			}
			arguments = append(arguments, test.command...)
			parsed, err := parseOptions(arguments)
			if err != nil {
				t.Fatal(err)
			}

			want := []string{
				fixture.interpreter,
				"-I",
				"-S",
				fixture.bootstrap,
				"--runtime-root",
				fixture.runtimeRoot,
				"--package-root",
				fixture.packageRoot,
				"--import-root",
				fixture.importRoot,
				"--",
			}
			want = append(want, test.command[1:]...)
			if !reflect.DeepEqual(parsed.command, want) {
				t.Fatalf(
					"bootstrap argv mismatch:\n got: %q\nwant: %q",
					parsed.command,
					want,
				)
			}

			count := 0
			for _, argument := range parsed.command {
				if argument == "--require-complete-source-unit" {
					count++
				}
			}
			if count != test.count {
				t.Fatalf(
					"complete-source-unit literal count = %d, want %d in %q",
					count,
					test.count,
					parsed.command,
				)
			}
		})
	}
}

func TestFixturePolicyRejectsSymlinkInPackageTree(t *testing.T) {
	temporary, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	packageRoot := filepath.Join(temporary, "axiom_encode")
	if err := os.Mkdir(packageRoot, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(packageRoot, "__init__.py"), []byte("\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("__init__.py", filepath.Join(packageRoot, "alias.py")); err != nil {
		t.Fatal(err)
	}
	if _, err := validateTrustedTree(packageRoot); err == nil {
		t.Fatal("expected package-tree symlink rejection")
	}
}

func TestFixturePolicyRejectsNestedShebangInterpreter(t *testing.T) {
	temporary, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	runtimeRoot := filepath.Join(temporary, "runtime")
	packageRoot := filepath.Join(runtimeRoot, "axiom_encode")
	if err := os.MkdirAll(packageRoot, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(packageRoot, "__init__.py"), []byte("\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	terminal := filepath.Join(runtimeRoot, "python3.13")
	if err := os.WriteFile(terminal, []byte{0x7f, 'E', 'L', 'F'}, 0o700); err != nil {
		t.Fatal(err)
	}
	wrapper := filepath.Join(runtimeRoot, "python-wrapper")
	if err := os.WriteFile(wrapper, []byte("#!"+terminal+"\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	executable := filepath.Join(runtimeRoot, "axiom-encode")
	if err := os.WriteFile(executable, []byte("#!"+wrapper+" -I\n"), 0o700); err != nil {
		t.Fatal(err)
	}

	if _, err := validateTrustedExecutionChain(
		executable,
		[]string{runtimeRoot},
		[]string{filepath.Dir(packageRoot)},
		packageRoot,
	); err == nil {
		t.Fatal("expected nested shebang interpreter rejection")
	}
}

func TestFixturePolicyRejectsPackageOutsideDeclaredRuntime(t *testing.T) {
	temporary, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	runtimeRoot := filepath.Join(temporary, "runtime")
	packageRoot := filepath.Join(temporary, "packages", "axiom_encode")
	if err := os.MkdirAll(filepath.Join(runtimeRoot, "bin"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(packageRoot, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(packageRoot, "__init__.py"), []byte("\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	interpreter := filepath.Join(runtimeRoot, "bin", "python3.13")
	if err := os.WriteFile(interpreter, []byte{0x7f, 'E', 'L', 'F'}, 0o700); err != nil {
		t.Fatal(err)
	}
	executable := filepath.Join(runtimeRoot, "axiom-encode")
	if err := os.WriteFile(executable, []byte("#!"+interpreter+" -I\n"), 0o700); err != nil {
		t.Fatal(err)
	}

	if _, err := validateTrustedExecutionChain(
		executable,
		[]string{runtimeRoot},
		[]string{filepath.Dir(packageRoot)},
		packageRoot,
	); err == nil {
		t.Fatal("expected package outside declared runtime rejection")
	}
}
