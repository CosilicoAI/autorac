//go:build darwin || linux

// axiom-encode-signing-supervisor is the compiled trust boundary for signed
// Axiom Encode operations. Private keys remain inside external signers. The
// supervisor accepts only operation-scoped signer sockets, provisions a separate
// broker over an anonymous socketpair, then replaces itself with a validated
// Python interpreter running the protected pre-attachment bootstrap. Neither raw
// private key bytes nor private-key environment variables enter Go or Python.
package main

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"encoding/pem"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"golang.org/x/sys/unix"
)

const (
	protocolVersion        = 4
	signerProtocolVersion  = 2
	maxFrameBytes          = 64 * 1024 * 1024
	maxJSONDepth           = 32
	maxJSONKeyBytes        = 256
	brokerServerFD         = 3
	signerChallengeBytes   = 32
	signerOperationTimeout = 10 * time.Second
	signerChallengeDomain  = "axiom-encode/external-signer-challenge/v2\x00"
	signerSignatureDomain  = "axiom-encode/external-signer-sign/v2\x00"

	legacyApplyPrivateEnv  = "AXIOM_ENCODE_APPLY_SIGNING_KEY"
	applyPrivateEnv        = "AXIOM_ENCODE_APPLY_SIGNING_PRIVATE_KEY"
	applyPublicEnv         = "AXIOM_ENCODE_APPLY_SIGNING_PUBLIC_KEY"
	evalPrivateEnv         = "AXIOM_ENCODE_EVAL_SIGNING_PRIVATE_KEY"
	evalPublicEnv          = "AXIOM_ENCODE_EVAL_SIGNING_PUBLIC_KEY"
	corpusReleasePublicEnv = "AXIOM_CORPUS_RELEASE_PUBLIC_KEY"

	brokerFDEnv     = "AXIOM_ENCODE_SIGNING_BROKER_FD"
	brokerPIDEnv    = "AXIOM_ENCODE_SIGNING_BROKER_PID"
	brokerActiveEnv = "AXIOM_ENCODE_SIGNING_BROKER_ACTIVE"
	// trustedRuntimeEnv marks the child as executing inside the root-provisioned,
	// git-free verification runtime. The encoder reads it to source its own apply
	// identity from the root-written runtime-attestation.json instead of a Git
	// checkout (which the provisioned runtime is not). It is set from the empty
	// child environment below, so it can never be spoofed by an ambient value.
	trustedRuntimeEnv      = "AXIOM_ENCODE_TRUSTED_RUNTIME"
	trustedCodexBinEnv     = "AXIOM_ENCODE_TRUSTED_CODEX_BIN"
	trustedCodexVersionEnv = "AXIOM_ENCODE_TRUSTED_CODEX_VERSION"
	trustedCodexSHA256Env  = "AXIOM_ENCODE_TRUSTED_CODEX_SHA256"
)

var privateEnvironmentNames = map[string]struct{}{
	legacyApplyPrivateEnv: {},
	applyPrivateEnv:       {},
	evalPrivateEnv:        {},
}

var publicEnvironmentNames = map[string]struct{}{
	applyPublicEnv:         {},
	evalPublicEnv:          {},
	corpusReleasePublicEnv: {},
}

var parentOnlyEnvironmentNames = []string{
	"OPENAI_API_KEY",
	"ANTHROPIC_API_KEY",
	"AXIOM_ENCODE_APPLY_CHECKOUT",
	"AXIOM_ENCODE_SUPABASE_URL",
	"AXIOM_ENCODE_SUPABASE_SECRET_KEY",
	"AXIOM_ENCODE_SUPABASE_ANON_KEY",
	"GITHUB_ACTIONS",
	"GITHUB_SHA",
	"GITHUB_WORKSPACE",
	"OTEL_EXPORTER_OTLP_ENDPOINT",
	"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
	"OTEL_EXPORTER_OTLP_HEADERS",
	"OTEL_SERVICE_NAME",
}

var brokerEnvironmentNames = map[string]struct{}{
	brokerFDEnv:     {},
	brokerPIDEnv:    {},
	brokerActiveEnv: {},
}

type options struct {
	applySignerFD      int
	evalSignerFD       int
	trustRootsPath     string
	pythonRuntimeRoots []string
	pythonImportRoots  []string
	pythonPackageRoot  string
	codexAuthPath      string
	codexAuthOutbox    string
	codexCLIConfigPath string
	command            []string
}

type trustedCodexCLI struct {
	Schema  string `json:"schema"`
	Version string `json:"version"`
	SHA256  string `json:"sha256"`
	Path    string `json:"path"`
}

type brokerOptions struct {
	applySignerFD int
	evalSignerFD  int
}

type signingTrustRoots struct {
	Schema                  string   `json:"schema"`
	ApplyPublicKey          string   `json:"apply_ed25519_public_key"`
	EvalPublicKey           string   `json:"eval_ed25519_public_key"`
	CorpusReleasePublicKey  *string  `json:"corpus_release_ed25519_public_key,omitempty"`
	CorpusReleasePublicKeys []string `json:"corpus_release_ed25519_public_keys,omitempty"`
}

type brokerRequest struct {
	Version                 int      `json:"version"`
	ID                      int64    `json:"id"`
	Operation               string   `json:"op"`
	Payload                 []byte   `json:"payload,omitempty"`
	ApplyPublicKey          []byte   `json:"apply_public_key,omitempty"`
	EvalPublicKey           []byte   `json:"eval_public_key,omitempty"`
	CorpusReleasePublicKey  []byte   `json:"corpus_release_public_key,omitempty"`
	CorpusReleasePublicKeys [][]byte `json:"corpus_release_public_keys,omitempty"`
	presentFields           map[string]struct{}
}

var brokerRequestFields = map[string]struct{}{
	"version":                    {},
	"id":                         {},
	"op":                         {},
	"payload":                    {},
	"apply_public_key":           {},
	"eval_public_key":            {},
	"corpus_release_public_key":  {},
	"corpus_release_public_keys": {},
}

func (request *brokerRequest) UnmarshalJSON(raw []byte) error {
	var encodedFields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &encodedFields); err != nil {
		return err
	}
	if encodedFields == nil {
		return errors.New("signing broker request must be a JSON object")
	}
	defer func() {
		for _, encoded := range encodedFields {
			zero(encoded)
		}
	}()
	for name := range encodedFields {
		if _, allowed := brokerRequestFields[name]; !allowed {
			return fmt.Errorf("json: unknown field %q", name)
		}
	}

	// Decode through a method-free alias after recording field presence. This
	// distinguishes omitted fields from explicitly empty or null fields, which
	// is necessary for an exact, non-ambiguous capability protocol.
	type wireBrokerRequest brokerRequest
	var decoded wireBrokerRequest
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return err
	}
	*request = brokerRequest(decoded)
	request.presentFields = make(map[string]struct{}, len(encodedFields))
	for name := range encodedFields {
		request.presentFields[name] = struct{}{}
	}
	return nil
}

func (request *brokerRequest) hasField(name string) bool {
	_, present := request.presentFields[name]
	return present
}

func (request *brokerRequest) hasExactFields(expected ...string) bool {
	if len(request.presentFields) != len(expected) {
		return false
	}
	for _, name := range expected {
		if !request.hasField(name) {
			return false
		}
	}
	return true
}

func (request *brokerRequest) hasInitializationFields() bool {
	return request.hasField("apply_public_key") ||
		request.hasField("eval_public_key") ||
		request.hasField("corpus_release_public_key") ||
		request.hasField("corpus_release_public_keys")
}

func (request *brokerRequest) hasValidInitializationShape() bool {
	if len(request.ApplyPublicKey) != ed25519.PublicKeySize ||
		len(request.EvalPublicKey) != ed25519.PublicKeySize ||
		len(request.CorpusReleasePublicKey) != ed25519.PublicKeySize ||
		len(request.CorpusReleasePublicKeys) == 0 ||
		!bytes.Equal(request.CorpusReleasePublicKey, request.CorpusReleasePublicKeys[0]) ||
		bytes.Equal(request.ApplyPublicKey, request.EvalPublicKey) {
		return false
	}
	seen := map[string]struct{}{
		string(request.ApplyPublicKey): {},
		string(request.EvalPublicKey):  {},
	}
	for _, publicKey := range request.CorpusReleasePublicKeys {
		if len(publicKey) != ed25519.PublicKeySize {
			return false
		}
		encoded := string(publicKey)
		if _, duplicate := seen[encoded]; duplicate {
			return false
		}
		seen[encoded] = struct{}{}
	}
	return len(request.ApplyPublicKey) == ed25519.PublicKeySize &&
		len(request.EvalPublicKey) == ed25519.PublicKeySize &&
		len(request.CorpusReleasePublicKey) == ed25519.PublicKeySize &&
		request.hasExactFields(
			"version", "id", "op", "apply_public_key", "eval_public_key",
			"corpus_release_public_key", "corpus_release_public_keys",
		)
}

func validateBrokerRequestEnvelope(request *brokerRequest, lastRequestID *int64) error {
	if request.Version != protocolVersion ||
		!request.hasField("version") ||
		!request.hasField("id") ||
		!request.hasField("op") {
		return errors.New("Signing broker request is malformed")
	}
	if request.ID <= 0 || request.ID <= *lastRequestID {
		return errors.New(
			"Signing broker request ID must be positive and strictly increasing",
		)
	}
	*lastRequestID = request.ID
	return nil
}

func validateBrokerRequestOperation(request *brokerRequest) error {
	if request.hasInitializationFields() {
		return errors.New("Signing broker request contains forbidden initialization fields")
	}
	switch request.Operation {
	case "shutdown", "status":
		if !request.hasExactFields("version", "id", "op") {
			return fmt.Errorf("Signing broker %s request is malformed", request.Operation)
		}
	case "apply_ed25519_sign", "eval_ed25519_sign":
		if !request.hasExactFields("version", "id", "op", "payload") || request.Payload == nil {
			return errors.New("Signing broker signing request is malformed")
		}
	default:
		if !request.hasExactFields("version", "id", "op") {
			return errors.New("Signing broker request is malformed")
		}
		return errors.New("Signing broker operation is not supported")
	}
	return nil
}

type brokerStatus struct {
	Capabilities            []string `json:"capabilities"`
	ApplyPublicKey          []byte   `json:"apply_public_key"`
	EvalPublicKey           []byte   `json:"eval_public_key"`
	CorpusReleasePublicKey  []byte   `json:"corpus_release_public_key"`
	CorpusReleasePublicKeys [][]byte `json:"corpus_release_public_keys"`
}

type brokerResponse struct {
	Version int    `json:"version"`
	ID      int64  `json:"id"`
	OK      bool   `json:"ok"`
	Error   string `json:"error,omitempty"`
	Result  any    `json:"result"`
}

type receivedBrokerResponse struct {
	Version int          `json:"version"`
	ID      int64        `json:"id"`
	OK      bool         `json:"ok"`
	Error   string       `json:"error,omitempty"`
	Result  brokerResult `json:"result"`
}

type brokerResult struct {
	Capabilities            []string `json:"capabilities,omitempty"`
	ApplyPublicKey          []byte   `json:"apply_public_key,omitempty"`
	EvalPublicKey           []byte   `json:"eval_public_key,omitempty"`
	CorpusReleasePublicKey  []byte   `json:"corpus_release_public_key,omitempty"`
	CorpusReleasePublicKeys [][]byte `json:"corpus_release_public_keys,omitempty"`
	Signature               []byte   `json:"signature,omitempty"`
}

type signerRequest struct {
	Version   int    `json:"version"`
	ID        int64  `json:"id"`
	Operation string `json:"op"`
	Scope     string `json:"scope"`
	Challenge []byte `json:"challenge,omitempty"`
	Payload   []byte `json:"payload,omitempty"`
}

type signerResponse struct {
	Version       int    `json:"version"`
	ID            int64  `json:"id"`
	OK            bool   `json:"ok"`
	Error         string `json:"error,omitempty"`
	PublicKey     []byte `json:"public_key,omitempty"`
	Signature     []byte `json:"signature,omitempty"`
	presentFields map[string]struct{}
}

var signerResponseFields = map[string]struct{}{
	"version":    {},
	"id":         {},
	"ok":         {},
	"error":      {},
	"public_key": {},
	"signature":  {},
}

func (response *signerResponse) UnmarshalJSON(raw []byte) error {
	var encodedFields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &encodedFields); err != nil {
		return err
	}
	if encodedFields == nil {
		return errors.New("external signer response must be a JSON object")
	}
	defer func() {
		for _, encoded := range encodedFields {
			zero(encoded)
		}
	}()
	for name := range encodedFields {
		if _, allowed := signerResponseFields[name]; !allowed {
			return fmt.Errorf("json: unknown field %q", name)
		}
	}
	type wireSignerResponse signerResponse
	var decoded wireSignerResponse
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return err
	}
	*response = signerResponse(decoded)
	response.presentFields = make(map[string]struct{}, len(encodedFields))
	for name := range encodedFields {
		response.presentFields[name] = struct{}{}
	}
	return nil
}

func (response *signerResponse) hasExactFields(expected ...string) bool {
	if len(response.presentFields) != len(expected) {
		return false
	}
	for _, name := range expected {
		if _, present := response.presentFields[name]; !present {
			return false
		}
	}
	return true
}

type externalSigner struct {
	label      string
	scope      string
	connection net.Conn
	publicKey  ed25519.PublicKey
	nextID     int64
}

func main() {
	if len(os.Args) == 2 && os.Args[1] == "--build-kind" {
		fmt.Println(supervisorBuildKind)
		return
	}
	if len(os.Args) >= 2 && os.Args[1] == "__broker" {
		parsed, err := parseBrokerOptions(os.Args[2:])
		if err == nil {
			err = serveBroker(brokerServerFD, parsed)
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "signing broker: %v\n", err)
			os.Exit(2)
		}
		return
	}
	if err := supervise(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "signing supervisor: %v\n", err)
		os.Exit(2)
	}
}

func parseOptions(arguments []string) (options, error) {
	parsed := options{applySignerFD: -1, evalSignerFD: -1}
	flags := flag.NewFlagSet("axiom-encode-signing-supervisor", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	flags.IntVar(
		&parsed.applySignerFD,
		"apply-signer-fd",
		-1,
		"inherited socket descriptor for the operation-scoped external apply signer",
	)
	flags.StringVar(
		&parsed.trustRootsPath,
		"trusted-signing-roots",
		"",
		"protected JSON file containing three distinct Ed25519 public trust roots",
	)
	flags.IntVar(
		&parsed.evalSignerFD,
		"eval-signer-fd",
		-1,
		"inherited socket descriptor for the operation-scoped external eval signer",
	)
	flags.Func(
		"trusted-python-runtime-root",
		"the one absolute root-owned self-contained Python runtime tree",
		func(value string) error {
			parsed.pythonRuntimeRoots = append(parsed.pythonRuntimeRoots, value)
			return nil
		},
	)
	flags.Func(
		"trusted-python-import-root",
		"absolute protected Python import root inside the self-contained runtime; repeat as needed",
		func(value string) error {
			parsed.pythonImportRoots = append(parsed.pythonImportRoots, value)
			return nil
		},
	)
	flags.StringVar(
		&parsed.pythonPackageRoot,
		"trusted-python-package-root",
		"",
		"absolute root-owned axiom_encode package directory",
	)
	flags.StringVar(&parsed.codexAuthPath, "codex-subscription-auth", "", "operator auth.json copied into an isolated runtime CODEX_HOME")
	flags.StringVar(&parsed.codexAuthOutbox, "codex-auth-outbox", "", "credential-bearing path receiving refreshed auth.json at teardown")
	flags.StringVar(&parsed.codexCLIConfigPath, "trusted-codex-cli-config", "", "protected pinned Codex CLI config")
	if err := flags.Parse(arguments); err != nil {
		return options{}, err
	}
	parsed.command = flags.Args()
	if len(parsed.command) == 0 {
		return options{}, errors.New(
			"expected `-- axiom-encode <arguments>` after signing options",
		)
	}
	if parsed.trustRootsPath == "" {
		return options{}, errors.New("--trusted-signing-roots is required")
	}
	if parsed.codexAuthPath == "" && (parsed.codexAuthOutbox != "" || parsed.codexCLIConfigPath != "") {
		return options{}, errors.New("Codex outbox/config requires --codex-subscription-auth")
	}
	if parsed.codexAuthPath != "" && (parsed.codexAuthOutbox == "" || parsed.codexCLIConfigPath == "") {
		return options{}, errors.New("Codex subscription auth requires --codex-auth-outbox and --trusted-codex-cli-config")
	}
	if filepath.Base(parsed.command[0]) != "axiom-encode" {
		return options{}, errors.New(
			"the supervised executable must be named axiom-encode",
		)
	}
	trustedChain, err := validateTrustedExecutionChain(
		parsed.command[0],
		parsed.pythonRuntimeRoots,
		parsed.pythonImportRoots,
		parsed.pythonPackageRoot,
	)
	if err != nil {
		return options{}, err
	}
	if trustedChain.PythonInterpreter != "" {
		pythonArguments := []string{
			trustedChain.PythonInterpreter,
			"-I",
			"-S",
			trustedChain.PythonBootstrap,
			"--runtime-root",
			trustedChain.PythonRuntimeRoots[0],
			"--package-root",
			trustedChain.PythonPackageRoot,
		}
		for _, root := range trustedChain.PythonImportRoots {
			pythonArguments = append(pythonArguments, "--import-root", root)
		}
		pythonArguments = append(pythonArguments, "--")
		pythonArguments = append(pythonArguments, parsed.command[1:]...)
		parsed.command = pythonArguments
	} else {
		parsed.command[0] = trustedChain.Executable
	}
	for label, descriptor := range map[string]int{
		"apply": parsed.applySignerFD,
		"eval":  parsed.evalSignerFD,
	} {
		if descriptor >= 0 && descriptor <= 2 {
			return options{}, fmt.Errorf(
				"%s signer descriptor must be greater than 2", label,
			)
		}
	}
	if parsed.applySignerFD >= 0 && parsed.applySignerFD == parsed.evalSignerFD {
		return options{}, errors.New("apply and eval signers need distinct descriptors")
	}
	for label, descriptor := range map[string]int{
		"apply": parsed.applySignerFD,
		"eval":  parsed.evalSignerFD,
	} {
		if descriptor >= 0 {
			if err := validateSignerDescriptor(descriptor); err != nil {
				return options{}, fmt.Errorf("%s signer descriptor is invalid: %w", label, err)
			}
		}
	}
	return parsed, nil
}

func parseBrokerOptions(arguments []string) (brokerOptions, error) {
	parsed := brokerOptions{applySignerFD: -1, evalSignerFD: -1}
	flags := flag.NewFlagSet("axiom-encode-signing-broker", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	flags.IntVar(&parsed.applySignerFD, "apply-signer-fd", -1, "internal apply signer descriptor")
	flags.IntVar(&parsed.evalSignerFD, "eval-signer-fd", -1, "internal eval signer descriptor")
	if err := flags.Parse(arguments); err != nil {
		return brokerOptions{}, err
	}
	if len(flags.Args()) != 0 {
		return brokerOptions{}, errors.New("unexpected internal broker arguments")
	}
	if parsed.applySignerFD >= 0 && parsed.applySignerFD == parsed.evalSignerFD {
		return brokerOptions{}, errors.New("broker signer descriptors must be distinct")
	}
	return parsed, nil
}

func supervise(arguments []string) error {
	if err := hardenProcess(); err != nil {
		return fmt.Errorf("could not harden supervisor process: %w", err)
	}
	for name := range privateEnvironmentNames {
		if _, present := os.LookupEnv(name); present {
			return fmt.Errorf(
				"private keys are forbidden in the signing supervisor; attach an external signer capability instead (forbidden environment variable %s)",
				name,
			)
		}
	}
	for name := range publicEnvironmentNames {
		if _, present := os.LookupEnv(name); present {
			return fmt.Errorf(
				"public signing roots must come from --trusted-signing-roots, not environment variable %s",
				name,
			)
		}
	}
	selfPath, err := os.Executable()
	if err != nil {
		return fmt.Errorf("could not resolve supervisor executable: %w", err)
	}
	if _, err := validateTrustedNativeExecutable(selfPath); err != nil {
		return fmt.Errorf("supervisor executable is not protected: %w", err)
	}
	parsed, err := parseOptions(arguments)
	if err != nil {
		return err
	}
	if parsed.codexAuthPath != "" {
		if value, present := os.LookupEnv("CODEX_HOME"); present && value != "" {
			return errors.New("ambient CODEX_HOME is outside supervisor custody; unset it and pass --codex-subscription-auth explicitly")
		}
	}

	applyPublicKey, evalPublicKey, corpusReleasePublicKeys, err := loadProtectedTrustRoots(
		parsed.trustRootsPath,
	)
	if err != nil {
		return err
	}

	connection, brokerProcess, err := startBroker(parsed)
	if err != nil {
		return err
	}
	cleanupBroker := true
	defer func() {
		if cleanupBroker {
			_ = connection.Close()
			_ = brokerProcess.Kill()
			_, _ = brokerProcess.Wait()
		}
	}()

	initialization := brokerRequest{
		Version:   protocolVersion,
		ID:        0,
		Operation: "initialize",
	}
	initialization.ApplyPublicKey = applyPublicKey
	initialization.EvalPublicKey = evalPublicKey
	initialization.CorpusReleasePublicKey = corpusReleasePublicKeys[0]
	initialization.CorpusReleasePublicKeys = corpusReleasePublicKeys
	if err := sendFrame(connection, initialization); err != nil {
		return fmt.Errorf("could not provision signing broker: %w", err)
	}
	var initialized receivedBrokerResponse
	if err := receiveFrame(connection, &initialized); err != nil {
		return fmt.Errorf("could not read signing broker initialization: %w", err)
	}
	if initialized.Version != protocolVersion || initialized.ID != 0 {
		return errors.New("signing broker initialization response is malformed")
	}
	if !initialized.OK {
		if initialized.Error == "" {
			initialized.Error = "broker rejected its initialization"
		}
		return errors.New(initialized.Error)
	}
	if err := validateStatus(
		initialized.Result,
		applyPublicKey,
		evalPublicKey,
		corpusReleasePublicKeys,
		parsed.applySignerFD >= 0,
		parsed.evalSignerFD >= 0,
	); err != nil {
		return err
	}

	capabilityFD := int(connection.Fd())
	trustedHome := filepath.Dir(parsed.command[0])
	if len(parsed.pythonRuntimeRoots) == 1 {
		trustedHome = parsed.pythonRuntimeRoots[0]
	}
	childEnvironment := cleanChildEnvironment(
		brokerProcess.Pid,
		capabilityFD,
		filepath.Dir(parsed.command[0]),
		trustedHome,
	)
	if parsed.codexAuthPath != "" {
		return superviseWithCodexSubscription(
			parsed, connection, childEnvironment, capabilityFD,
		)
	}

	// Clear close-on-exec only for the one-generation capability and seal every
	// other inherited descriptor. Retaining the socket's existing descriptor
	// avoids replacing a Go runtime kqueue/epoll descriptor immediately before
	// exec. Python marks the capability close-on-exec as soon as its entrypoint
	// attaches; model subprocesses therefore cannot inherit it.
	if err := makeInheritable(capabilityFD); err != nil {
		return fmt.Errorf("could not prepare signing capability: %w", err)
	}
	if err := sealOtherDescriptors(capabilityFD); err != nil {
		return fmt.Errorf("could not seal inherited descriptors: %w", err)
	}

	cleanupBroker = false
	if err := syscall.Exec(parsed.command[0], parsed.command, childEnvironment); err != nil {
		cleanupBroker = true
		return fmt.Errorf("could not execute trusted Python bootstrap: %w", err)
	}
	return nil
}

func loadTrustedCodexCLI(path string) (trustedCodexCLI, error) {
	_, file, err := inspectTrustedRegularFile(path, false)
	if err != nil {
		return trustedCodexCLI{}, fmt.Errorf("Codex CLI config is not protected: %w", err)
	}
	defer file.Close()
	raw, err := io.ReadAll(io.LimitReader(file, 16*1024+1))
	if err != nil || len(raw) > 16*1024 {
		return trustedCodexCLI{}, errors.New("could not read bounded Codex CLI config")
	}
	var config trustedCodexCLI
	if err := json.Unmarshal(raw, &config); err != nil {
		return trustedCodexCLI{}, fmt.Errorf("Codex CLI config is malformed: %w", err)
	}
	if config.Schema != "axiom-encode/trusted-codex-cli/v1" || config.Version == "" ||
		len(config.SHA256) != 64 || !filepath.IsAbs(config.Path) {
		return trustedCodexCLI{}, errors.New("Codex CLI config has invalid fields")
	}
	trustedPath, binary, err := inspectTrustedRegularFile(config.Path, true)
	if err != nil {
		return trustedCodexCLI{}, fmt.Errorf("pinned Codex CLI is not protected: %w", err)
	}
	defer binary.Close()
	digest := sha256.New()
	if _, err := io.Copy(digest, binary); err != nil {
		return trustedCodexCLI{}, fmt.Errorf("could not hash pinned Codex CLI: %w", err)
	}
	actual := fmt.Sprintf("%x", digest.Sum(nil))
	if !strings.EqualFold(actual, config.SHA256) {
		return trustedCodexCLI{}, fmt.Errorf("pinned Codex CLI sha256 mismatch: expected %s, got %s", config.SHA256, actual)
	}
	config.Path = trustedPath
	config.SHA256 = actual
	return config, nil
}

func copyCredential(sourcePath, destinationPath string, exclusive bool) (os.FileInfo, error) {
	flags := syscall.O_RDONLY
	if definedNoFollow() {
		flags |= syscall.O_NOFOLLOW
	}
	descriptor, err := syscall.Open(sourcePath, flags, 0)
	if err != nil {
		return nil, err
	}
	source := os.NewFile(uintptr(descriptor), sourcePath)
	defer source.Close()
	metadata, err := source.Stat()
	if err != nil || !metadata.Mode().IsRegular() || metadata.Size() > 1024*1024 {
		return nil, errors.New("Codex auth source must be a regular file no larger than 1 MiB")
	}
	destinationFlags := os.O_WRONLY | os.O_CREATE
	if exclusive {
		destinationFlags |= os.O_EXCL
	} else {
		destinationFlags |= os.O_TRUNC
	}
	destination, err := os.OpenFile(destinationPath, destinationFlags, 0600)
	if err != nil {
		return nil, err
	}
	_, copyErr := io.Copy(destination, source)
	syncErr := destination.Sync()
	closeErr := destination.Close()
	if copyErr != nil {
		return nil, copyErr
	}
	if syncErr != nil {
		return nil, syncErr
	}
	if closeErr != nil {
		return nil, closeErr
	}
	return metadata, nil
}

func definedNoFollow() bool { return syscall.O_NOFOLLOW != 0 }

func validateCodexScratchPolicy(mode os.FileMode, ownerUID int, runtimeUID int) error {
	if !mode.IsDir() || mode.Perm() != 0700 {
		return errors.New("Codex scratch home must be a protected 0700 directory")
	}
	if ownerUID != runtimeUID {
		return errors.New("Codex scratch home is not runtime-owned")
	}
	return nil
}

func openCredentialOutboxDirectory(outboxPath string) (int, string, error) {
	if !filepath.IsAbs(outboxPath) || filepath.Clean(outboxPath) != outboxPath || filepath.Base(outboxPath) == "." {
		return -1, "", errors.New("Codex auth outbox must be an absolute canonical file path")
	}
	name := filepath.Base(outboxPath)
	fd, err := unix.Open("/", unix.O_RDONLY|unix.O_DIRECTORY|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err != nil {
		return -1, "", fmt.Errorf("could not open filesystem root: %w", err)
	}
	for _, component := range strings.Split(strings.TrimPrefix(filepath.Dir(outboxPath), "/"), "/") {
		if component == "" {
			continue
		}
		next, openErr := unix.Openat(fd, component, unix.O_RDONLY|unix.O_DIRECTORY|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
		_ = unix.Close(fd)
		if openErr != nil {
			return -1, "", fmt.Errorf("Codex auth outbox directory must not contain symlinks: %w", openErr)
		}
		fd = next
	}
	var directoryStat unix.Stat_t
	if err := unix.Fstat(fd, &directoryStat); err != nil || directoryStat.Mode&unix.S_IFMT != unix.S_IFDIR || directoryStat.Mode&0022 != 0 || int(directoryStat.Uid) != os.Geteuid() {
		_ = unix.Close(fd)
		return -1, "", errors.New("Codex auth outbox directory must be operator-owned and not group/other-writable")
	}
	return fd, name, nil
}

func validateCredentialOutbox(outboxPath string) error {
	directoryFD, name, err := openCredentialOutboxDirectory(outboxPath)
	if err != nil {
		return err
	}
	defer unix.Close(directoryFD)
	if err := validateCredentialOutboxDestination(directoryFD, name); err != nil {
		return err
	}
	return nil
}

func validateCredentialOutboxDestination(directoryFD int, name string) error {
	var destinationStat unix.Stat_t
	err := unix.Fstatat(directoryFD, name, &destinationStat, unix.AT_SYMLINK_NOFOLLOW)
	if errors.Is(err, unix.ENOENT) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("could not inspect Codex auth outbox: %w", err)
	}
	if destinationStat.Mode&unix.S_IFMT != unix.S_IFREG {
		return errors.New("Codex auth outbox must not be a symlink or special file")
	}
	if int(destinationStat.Uid) != os.Geteuid() {
		return errors.New("existing Codex auth outbox must be operator-owned")
	}
	return nil
}

func publishCredential(sourcePath, outboxPath string, owner os.FileInfo) error {
	directoryFD, destinationName, err := openCredentialOutboxDirectory(outboxPath)
	if err != nil {
		return err
	}
	defer unix.Close(directoryFD)
	var randomBytes [16]byte
	if _, err := rand.Read(randomBytes[:]); err != nil {
		return fmt.Errorf("could not generate Codex auth temporary name: %w", err)
	}
	temporaryName := fmt.Sprintf(".%s.%x.tmp", destinationName, randomBytes[:])
	temporaryFD, err := unix.Openat(directoryFD, temporaryName, unix.O_WRONLY|unix.O_CREAT|unix.O_EXCL|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0600)
	if err != nil {
		return fmt.Errorf("could not create Codex auth temporary: %w", err)
	}
	temporary := os.NewFile(uintptr(temporaryFD), temporaryName)
	removeTemporary := true
	defer func() {
		if removeTemporary {
			_ = unix.Unlinkat(directoryFD, temporaryName, 0)
		}
	}()
	sourceFlags := syscall.O_RDONLY | syscall.O_NOFOLLOW
	sourceFD, err := syscall.Open(sourcePath, sourceFlags, 0)
	if err != nil {
		temporary.Close()
		return err
	}
	source := os.NewFile(uintptr(sourceFD), sourcePath)
	_, copyErr := io.Copy(temporary, io.LimitReader(source, 1024*1024+1))
	sourceCloseErr := source.Close()
	var chownErr error
	if stat, ok := owner.Sys().(*syscall.Stat_t); ok {
		chownErr = unix.Fchown(temporaryFD, int(stat.Uid), int(stat.Gid))
	}
	syncErr := temporary.Sync()
	closeErr := temporary.Close()
	if copyErr != nil || sourceCloseErr != nil || chownErr != nil || syncErr != nil || closeErr != nil {
		return errors.New("could not copy refreshed Codex auth to outbox")
	}
	if err := validateCredentialOutboxDestination(directoryFD, destinationName); err != nil {
		return err
	}
	if err := unix.Renameat(directoryFD, temporaryName, directoryFD, destinationName); err != nil {
		return fmt.Errorf("could not publish refreshed Codex auth outbox: %w", err)
	}
	removeTemporary = false
	return nil
}

func superviseWithCodexSubscription(parsed options, connection *os.File, environment []string, capabilityFD int) error {
	config, err := loadTrustedCodexCLI(parsed.codexCLIConfigPath)
	if err != nil {
		return err
	}
	if err := validateCredentialOutbox(parsed.codexAuthOutbox); err != nil {
		return err
	}
	home, err := os.MkdirTemp("", "axiom-codex-")
	if err != nil {
		return fmt.Errorf("could not create Codex scratch home: %w", err)
	}
	defer os.RemoveAll(home)
	if err := os.Chmod(home, 0700); err != nil {
		return err
	}
	homeMetadata, err := os.Lstat(home)
	if err != nil {
		return errors.New("could not inspect Codex scratch home")
	}
	homeStat, ok := homeMetadata.Sys().(*syscall.Stat_t)
	if !ok {
		return errors.New("could not inspect Codex scratch home ownership")
	}
	if err := validateCodexScratchPolicy(homeMetadata.Mode(), int(homeStat.Uid), os.Geteuid()); err != nil {
		return err
	}
	authPath := filepath.Join(home, "auth.json")
	authMetadata, err := copyCredential(parsed.codexAuthPath, authPath, true)
	if err != nil {
		return fmt.Errorf("could not materialize Codex auth: %w", err)
	}
	if err := os.WriteFile(filepath.Join(home, "config.toml"), []byte("check_for_update_on_startup = false\n"), 0600); err != nil {
		return err
	}
	environment = append(
		environment,
		"CODEX_HOME="+home,
		trustedCodexBinEnv+"="+config.Path,
		trustedCodexVersionEnv+"="+config.Version,
		trustedCodexSHA256Env+"="+config.SHA256,
	)
	for index, entry := range environment {
		if strings.HasPrefix(entry, brokerFDEnv+"=") {
			environment[index] = brokerFDEnv + "=3"
		}
	}
	command := exec.Command(parsed.command[0], parsed.command[1:]...)
	command.Env = environment
	command.Stdin, command.Stdout, command.Stderr = os.Stdin, os.Stdout, os.Stderr
	command.ExtraFiles = []*os.File{connection}
	if err := sealOtherDescriptors(capabilityFD); err != nil {
		return fmt.Errorf("could not seal inherited descriptors: %w", err)
	}
	commandErr := command.Run()
	refreshed := filepath.Join(home, "auth.json")
	if err := publishCredential(refreshed, parsed.codexAuthOutbox, authMetadata); err != nil {
		return err
	}
	if commandErr != nil {
		return fmt.Errorf("trusted generation failed: %w", commandErr)
	}
	return nil
}

func loadProtectedTrustRoots(path string) ([]byte, []byte, [][]byte, error) {
	trustedPath, file, err := inspectTrustedRegularFile(path, false)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("signing trust-root config is not protected: %w", err)
	}
	defer file.Close()
	raw, err := io.ReadAll(io.LimitReader(file, 64*1024+1))
	if err != nil {
		return nil, nil, nil, fmt.Errorf("could not read signing trust-root config: %w", err)
	}
	defer zero(raw)
	if len(raw) > 64*1024 {
		return nil, nil, nil, errors.New("signing trust-root config exceeds 64 KiB")
	}
	if err := rejectDuplicateJSONKeys(raw); err != nil {
		return nil, nil, nil, fmt.Errorf("signing trust-root config is malformed: %w", err)
	}
	return parseProtectedTrustRoots(raw, trustedPath)
}

func parseProtectedTrustRoots(raw []byte, trustedPath string) ([]byte, []byte, [][]byte, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil || fields == nil {
		return nil, nil, nil, errors.New("signing trust-root config must be one JSON object")
	}
	allowed := map[string]struct{}{
		"schema":                             {},
		"apply_ed25519_public_key":           {},
		"eval_ed25519_public_key":            {},
		"corpus_release_ed25519_public_key":  {},
		"corpus_release_ed25519_public_keys": {},
	}
	for name := range fields {
		if _, ok := allowed[name]; !ok {
			return nil, nil, nil, fmt.Errorf("signing trust-root config has unknown field %q", name)
		}
	}
	var config signingTrustRoots
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&config); err != nil {
		return nil, nil, nil, fmt.Errorf("signing trust-root config is malformed: %w", err)
	}
	const schemaV2 = "axiom-encode/signing-trust-roots/v2"
	const schemaV3 = "axiom-encode/signing-trust-roots/v3"
	if config.Schema != schemaV2 && config.Schema != schemaV3 {
		return nil, nil, nil, errors.New("signing trust-root config schema is unsupported")
	}
	for _, required := range []string{
		"schema", "apply_ed25519_public_key", "eval_ed25519_public_key",
	} {
		if _, present := fields[required]; !present {
			return nil, nil, nil, errors.New("signing trust-root config has the wrong fields")
		}
	}
	_, hasCorpusReleasePublicKey := fields["corpus_release_ed25519_public_key"]
	_, hasCorpusReleasePublicKeys := fields["corpus_release_ed25519_public_keys"]
	if config.Schema == schemaV2 {
		if len(fields) != 4 || !hasCorpusReleasePublicKey || hasCorpusReleasePublicKeys {
			return nil, nil, nil, errors.New("signing trust-root config has the wrong fields")
		}
	} else if (!hasCorpusReleasePublicKey && !hasCorpusReleasePublicKeys) || len(fields) > 5 {
		return nil, nil, nil, errors.New("signing trust-root config has the wrong fields")
	}
	applyPublicKey, err := parsePublicKey([]byte(config.ApplyPublicKey))
	if err != nil {
		return nil, nil, nil, fmt.Errorf("invalid apply manifest public key in %s: %w", trustedPath, err)
	}
	evalPublicKey, err := parsePublicKey([]byte(config.EvalPublicKey))
	if err != nil {
		zero(applyPublicKey)
		return nil, nil, nil, fmt.Errorf("invalid eval evidence public key in %s: %w", trustedPath, err)
	}
	var corpusReleasePublicKeys [][]byte
	if hasCorpusReleasePublicKeys {
		if len(config.CorpusReleasePublicKeys) == 0 {
			zero(applyPublicKey)
			zero(evalPublicKey)
			return nil, nil, nil, errors.New("corpus release public keyring must not be empty")
		}
		for index, encoded := range config.CorpusReleasePublicKeys {
			publicKey, parseErr := parseRawPublicKey([]byte(encoded))
			if parseErr != nil {
				zero(applyPublicKey)
				zero(evalPublicKey)
				zeroPublicKeys(corpusReleasePublicKeys)
				return nil, nil, nil, fmt.Errorf(
					"invalid corpus release public key %d in %s: %w",
					index, trustedPath, parseErr,
				)
			}
			corpusReleasePublicKeys = append(corpusReleasePublicKeys, publicKey)
		}
	} else {
		if config.CorpusReleasePublicKey == nil {
			zero(applyPublicKey)
			zero(evalPublicKey)
			return nil, nil, nil, errors.New("corpus release public key is missing")
		}
		publicKey, parseErr := parsePublicKey([]byte(*config.CorpusReleasePublicKey))
		if parseErr != nil {
			zero(applyPublicKey)
			zero(evalPublicKey)
			return nil, nil, nil, fmt.Errorf(
				"invalid corpus release public key in %s: %w", trustedPath, parseErr,
			)
		}
		corpusReleasePublicKeys = [][]byte{publicKey}
	}
	if hasCorpusReleasePublicKey && hasCorpusReleasePublicKeys {
		if config.CorpusReleasePublicKey == nil {
			zero(applyPublicKey)
			zero(evalPublicKey)
			zeroPublicKeys(corpusReleasePublicKeys)
			return nil, nil, nil, errors.New("corpus release public key is missing")
		}
		singularPublicKey, parseErr := parsePublicKey([]byte(*config.CorpusReleasePublicKey))
		if parseErr != nil {
			zero(applyPublicKey)
			zero(evalPublicKey)
			zeroPublicKeys(corpusReleasePublicKeys)
			return nil, nil, nil, fmt.Errorf(
				"invalid corpus release public key in %s: %w", trustedPath, parseErr,
			)
		}
		matchesCurrent := bytes.Equal(singularPublicKey, corpusReleasePublicKeys[0])
		zero(singularPublicKey)
		if !matchesCurrent {
			zero(applyPublicKey)
			zero(evalPublicKey)
			zeroPublicKeys(corpusReleasePublicKeys)
			return nil, nil, nil, errors.New(
				"singular and plural corpus release public keys conflict",
			)
		}
	}
	if bytes.Equal(applyPublicKey, evalPublicKey) {
		zero(applyPublicKey)
		zero(evalPublicKey)
		zeroPublicKeys(corpusReleasePublicKeys)
		return nil, nil, nil, errors.New(
			"apply, eval, and corpus release trust roots must be distinct",
		)
	}
	seen := map[string]struct{}{
		string(applyPublicKey): {},
		string(evalPublicKey):  {},
	}
	for _, publicKey := range corpusReleasePublicKeys {
		encoded := string(publicKey)
		if _, duplicate := seen[encoded]; duplicate {
			zero(applyPublicKey)
			zero(evalPublicKey)
			zeroPublicKeys(corpusReleasePublicKeys)
			return nil, nil, nil, errors.New(
				"apply, eval, and corpus release trust roots must be distinct",
			)
		}
		seen[encoded] = struct{}{}
	}
	return applyPublicKey, evalPublicKey, corpusReleasePublicKeys, nil
}

func parseRawPublicKey(material []byte) ([]byte, error) {
	normalized := bytes.TrimSpace(material)
	decoded := make([]byte, base64.StdEncoding.DecodedLen(len(normalized)))
	count, err := base64.StdEncoding.Strict().Decode(decoded, normalized)
	if err != nil {
		zero(decoded)
		return nil, errors.New("key must be base64-encoded raw bytes")
	}
	decoded = decoded[:count]
	if len(decoded) != ed25519.PublicKeySize {
		zero(decoded)
		return nil, fmt.Errorf("Ed25519 public key must contain %d raw bytes", ed25519.PublicKeySize)
	}
	return decoded, nil
}

func zeroPublicKeys(publicKeys [][]byte) {
	for _, publicKey := range publicKeys {
		zero(publicKey)
	}
}

func parsePublicKey(material []byte) ([]byte, error) {
	normalized := normalizeKeyMaterial(material)
	defer zero(normalized)
	if block, rest := pem.Decode(normalized); block != nil {
		defer zero(block.Bytes)
		if len(bytes.TrimSpace(rest)) != 0 {
			return nil, errors.New("PEM contains trailing data")
		}
		parsed, err := x509.ParsePKIXPublicKey(block.Bytes)
		if err != nil {
			return nil, errors.New("PEM must contain a PKIX public key")
		}
		publicKey, ok := parsed.(ed25519.PublicKey)
		if !ok {
			return nil, errors.New("key must be Ed25519")
		}
		return append([]byte(nil), publicKey...), nil
	}
	decoded := make([]byte, base64.StdEncoding.DecodedLen(len(normalized)))
	count, err := base64.StdEncoding.Strict().Decode(decoded, normalized)
	if err != nil {
		zero(decoded)
		return nil, errors.New("key must be PKIX PEM or base64-encoded raw bytes")
	}
	decoded = decoded[:count]
	if len(decoded) != ed25519.PublicKeySize {
		zero(decoded)
		return nil, fmt.Errorf("Ed25519 public key must contain %d raw bytes", ed25519.PublicKeySize)
	}
	return decoded, nil
}

func normalizeKeyMaterial(material []byte) []byte {
	normalized := bytes.TrimSpace(material)
	normalized = bytes.ReplaceAll(normalized, []byte(`\n`), []byte("\n"))
	return append([]byte(nil), normalized...)
}

func validateSignerDescriptor(descriptor int) error {
	if descriptor <= 2 {
		return errors.New("descriptor must be greater than 2")
	}
	var metadata syscall.Stat_t
	if err := syscall.Fstat(descriptor, &metadata); err != nil {
		return err
	}
	if metadata.Mode&syscall.S_IFMT != syscall.S_IFSOCK {
		return errors.New("descriptor must be a connected socket")
	}
	socketType, err := syscall.GetsockoptInt(descriptor, syscall.SOL_SOCKET, syscall.SO_TYPE)
	if err != nil {
		return err
	}
	if socketType != syscall.SOCK_STREAM {
		return errors.New("descriptor must be a stream socket")
	}
	peer, err := syscall.Getpeername(descriptor)
	if err != nil {
		return fmt.Errorf("descriptor must be connected: %w", err)
	}
	if _, ok := peer.(*syscall.SockaddrUnix); !ok {
		return errors.New("descriptor must be a local Unix socket")
	}
	syscall.CloseOnExec(descriptor)
	return nil
}

func startBroker(parsed options) (*os.File, *os.Process, error) {
	descriptors, err := syscall.Socketpair(syscall.AF_UNIX, syscall.SOCK_STREAM, 0)
	if err != nil {
		return nil, nil, fmt.Errorf("could not create anonymous broker socket: %w", err)
	}
	syscall.CloseOnExec(descriptors[0])
	syscall.CloseOnExec(descriptors[1])
	client := os.NewFile(uintptr(descriptors[0]), "signing-client")
	server := os.NewFile(uintptr(descriptors[1]), "signing-server")
	if client == nil || server == nil {
		if client != nil {
			_ = client.Close()
		}
		if server != nil {
			_ = server.Close()
		}
		return nil, nil, errors.New("could not wrap anonymous broker socket")
	}
	executable, err := os.Executable()
	if err != nil {
		_ = client.Close()
		_ = server.Close()
		return nil, nil, fmt.Errorf("could not resolve supervisor executable: %w", err)
	}
	trustedSupervisor, err := validateTrustedNativeExecutable(executable)
	if err != nil {
		_ = client.Close()
		_ = server.Close()
		return nil, nil, fmt.Errorf("supervisor executable is not protected: %w", err)
	}
	brokerArguments := []string{"__broker"}
	extraFiles := []*os.File{server}
	signerFiles := make([]*os.File, 0, 2)
	appendSigner := func(label string, descriptor int) error {
		if descriptor < 0 {
			return nil
		}
		file := os.NewFile(uintptr(descriptor), label+"-external-signer")
		if file == nil {
			return fmt.Errorf("could not wrap %s external signer descriptor", label)
		}
		childDescriptor := 3 + len(extraFiles)
		brokerArguments = append(
			brokerArguments,
			"--"+label+"-signer-fd",
			strconv.Itoa(childDescriptor),
		)
		extraFiles = append(extraFiles, file)
		signerFiles = append(signerFiles, file)
		return nil
	}
	if err := appendSigner("apply", parsed.applySignerFD); err != nil {
		_ = client.Close()
		_ = server.Close()
		return nil, nil, err
	}
	if err := appendSigner("eval", parsed.evalSignerFD); err != nil {
		_ = client.Close()
		_ = server.Close()
		for _, file := range signerFiles {
			_ = file.Close()
		}
		return nil, nil, err
	}
	command := exec.Command(trustedSupervisor, brokerArguments...)
	command.Env = []string{}
	command.ExtraFiles = extraFiles
	command.Stderr = os.Stderr
	if err := command.Start(); err != nil {
		_ = client.Close()
		_ = server.Close()
		for _, file := range signerFiles {
			_ = file.Close()
		}
		return nil, nil, fmt.Errorf("could not start signing broker: %w", err)
	}
	_ = server.Close()
	for _, file := range signerFiles {
		_ = file.Close()
	}
	return client, command.Process, nil
}

func validateStatus(
	result brokerResult,
	applyPublicKey, evalPublicKey []byte,
	corpusReleasePublicKeys [][]byte,
	hasApplyCapability, hasEvalCapability bool,
) error {
	expectedCapabilities := make([]string, 0, 2)
	if hasApplyCapability {
		expectedCapabilities = append(expectedCapabilities, "apply_ed25519")
	}
	if hasEvalCapability {
		expectedCapabilities = append(expectedCapabilities, "eval_ed25519")
	}
	if !bytes.Equal(result.ApplyPublicKey, applyPublicKey) {
		return errors.New("signing broker returned the wrong apply public key")
	}
	if !bytes.Equal(result.EvalPublicKey, evalPublicKey) {
		return errors.New("signing broker returned the wrong eval public key")
	}
	if !bytes.Equal(result.CorpusReleasePublicKey, corpusReleasePublicKeys[0]) {
		return errors.New("signing broker returned the wrong corpus release public key")
	}
	if !equalByteSlices(result.CorpusReleasePublicKeys, corpusReleasePublicKeys) {
		return errors.New("signing broker returned the wrong corpus release public keyring")
	}
	if !equalStrings(result.Capabilities, expectedCapabilities) {
		return errors.New("signing broker returned unexpected capabilities")
	}
	return nil
}

func cleanChildEnvironment(
	brokerPID int,
	capabilityFD int,
	trustedToolDirectory string,
	trustedHome string,
) []string {
	// Build from empty. No ambient PATH, Git configuration, proxy, cloud
	// credential, Supabase, or telemetry value crosses the trust boundary.
	clean := map[string]string{
		"GIT_CONFIG_GLOBAL":       "/dev/null",
		"GIT_CONFIG_NOSYSTEM":     "1",
		"GIT_TERMINAL_PROMPT":     "0",
		"HOME":                    trustedHome,
		"LANG":                    "C.UTF-8",
		"PATH":                    trustedToolDirectory,
		"PYTHONNOUSERSITE":        "1",
		"PYTHONSAFEPATH":          "1",
		"PYTHONUNBUFFERED":        "1",
		"PYTHONDONTWRITEBYTECODE": "1",
		"XDG_CONFIG_HOME":         filepath.Join(trustedHome, ".empty-config"),
		"XDG_DATA_HOME":           filepath.Join(trustedHome, ".empty-data"),
	}
	clean[brokerActiveEnv] = "1"
	clean[trustedRuntimeEnv] = "1"
	clean[brokerFDEnv] = strconv.Itoa(capabilityFD)
	clean[brokerPIDEnv] = strconv.Itoa(brokerPID)
	for _, name := range parentOnlyEnvironmentNames {
		if value, present := os.LookupEnv(name); present {
			clean[name] = value
		}
	}
	names := make([]string, 0, len(clean))
	for name := range clean {
		names = append(names, name)
	}
	// A stable environment order makes the exec input reproducible and easier to
	// audit. Manual insertion sort avoids adding a second helper dependency.
	for index := 1; index < len(names); index++ {
		for cursor := index; cursor > 0 && names[cursor] < names[cursor-1]; cursor-- {
			names[cursor], names[cursor-1] = names[cursor-1], names[cursor]
		}
	}
	environment := make([]string, 0, len(names))
	for _, name := range names {
		environment = append(environment, name+"="+clean[name])
	}
	return environment
}

func makeInheritable(descriptor int) error {
	if descriptor < 3 {
		return errors.New("signing capability descriptor must be greater than 2")
	}
	_, _, errno := syscall.Syscall(
		syscall.SYS_FCNTL,
		uintptr(descriptor),
		uintptr(syscall.F_SETFD),
		0,
	)
	if errno != 0 {
		return errno
	}
	return nil
}

func sealOtherDescriptors(keep int) error {
	directory := "/proc/self/fd"
	if _, err := os.Stat(directory); err != nil {
		directory = "/dev/fd"
	}
	descriptorDirectory, err := os.Open(directory)
	if err != nil {
		return err
	}
	names, readErr := descriptorDirectory.Readdirnames(-1)
	closeErr := descriptorDirectory.Close()
	if readErr != nil {
		return readErr
	}
	if closeErr != nil {
		return closeErr
	}
	for _, name := range names {
		descriptor, err := strconv.Atoi(name)
		if err != nil || descriptor <= 2 || descriptor == keep {
			continue
		}
		// Do not close Go's kqueue/epoll descriptors while the runtime is still
		// active. Mark every non-capability descriptor close-on-exec instead;
		// syscall.Exec then gives Python exactly stdio plus descriptor 3.
		syscall.CloseOnExec(descriptor)
	}
	return nil
}

func connectExternalSigner(
	label string,
	descriptor int,
	expectedPublicKey []byte,
) (*externalSigner, error) {
	if err := validateSignerDescriptor(descriptor); err != nil {
		return nil, err
	}
	file := os.NewFile(uintptr(descriptor), label+"-external-signer")
	if file == nil {
		return nil, errors.New("external signer descriptor is invalid")
	}
	connection, err := net.FileConn(file)
	_ = file.Close()
	if err != nil {
		return nil, fmt.Errorf("could not attach external signer socket: %w", err)
	}
	if _, ok := connection.(*net.UnixConn); !ok {
		_ = connection.Close()
		return nil, errors.New("external signer must use a Unix socket")
	}
	signer := &externalSigner{
		label:      label,
		scope:      label + "_ed25519",
		connection: connection,
		publicKey:  append(ed25519.PublicKey(nil), expectedPublicKey...),
		nextID:     1,
	}
	nonce := make([]byte, signerChallengeBytes)
	if _, err := rand.Read(nonce); err != nil {
		_ = signer.close()
		return nil, fmt.Errorf("could not create signer challenge: %w", err)
	}
	defer zero(nonce)
	challenge := make([]byte, 0, len(signerChallengeDomain)+len(signer.scope)+1+len(nonce))
	challenge = append(challenge, signerChallengeDomain...)
	challenge = append(challenge, signer.scope...)
	challenge = append(challenge, 0)
	challenge = append(challenge, nonce...)
	defer zero(challenge)
	response, err := signer.exchange(signerRequest{
		Version:   signerProtocolVersion,
		ID:        signer.nextID,
		Operation: "challenge",
		Scope:     signer.scope,
		Challenge: nonce,
	})
	signer.nextID++
	if err != nil {
		_ = signer.close()
		return nil, err
	}
	defer zero(response.PublicKey)
	defer zero(response.Signature)
	if !response.OK {
		_ = signer.close()
		return nil, fmt.Errorf("external %s signer rejected its challenge: %s", label, response.Error)
	}
	if !response.hasExactFields("version", "id", "ok", "public_key", "signature") ||
		!bytes.Equal(response.PublicKey, expectedPublicKey) ||
		len(response.Signature) != ed25519.SignatureSize ||
		!ed25519.Verify(signer.publicKey, challenge, response.Signature) {
		_ = signer.close()
		return nil, fmt.Errorf("external %s signer challenge response is invalid", label)
	}
	return signer, nil
}

func (signer *externalSigner) exchange(request signerRequest) (signerResponse, error) {
	if err := signer.connection.SetDeadline(time.Now().Add(signerOperationTimeout)); err != nil {
		return signerResponse{}, err
	}
	defer signer.connection.SetDeadline(time.Time{}) //nolint:errcheck
	if err := sendFrame(signer.connection, request); err != nil {
		return signerResponse{}, err
	}
	var response signerResponse
	if err := receiveFrame(signer.connection, &response); err != nil {
		return signerResponse{}, err
	}
	if response.Version != signerProtocolVersion || response.ID != request.ID {
		zero(response.PublicKey)
		zero(response.Signature)
		return signerResponse{}, errors.New("external signer response identity is invalid")
	}
	if !response.OK {
		if !response.hasExactFields("version", "id", "ok", "error") || response.Error == "" {
			return signerResponse{}, errors.New("external signer error response is malformed")
		}
	}
	return response, nil
}

func (signer *externalSigner) sign(payload []byte) ([]byte, error) {
	message := make([]byte, 0, len(signerSignatureDomain)+len(signer.scope)+1+len(payload))
	message = append(message, signerSignatureDomain...)
	message = append(message, signer.scope...)
	message = append(message, 0)
	message = append(message, payload...)
	defer zero(message)
	requestID := signer.nextID
	signer.nextID++
	response, err := signer.exchange(signerRequest{
		Version:   signerProtocolVersion,
		ID:        requestID,
		Operation: "sign",
		Scope:     signer.scope,
		Payload:   payload,
	})
	if err != nil {
		return nil, err
	}
	defer zero(response.PublicKey)
	defer zero(response.Signature)
	if !response.OK {
		return nil, fmt.Errorf("external %s signer rejected the request: %s", signer.label, response.Error)
	}
	if !response.hasExactFields("version", "id", "ok", "signature") ||
		len(response.Signature) != ed25519.SignatureSize ||
		!ed25519.Verify(signer.publicKey, message, response.Signature) {
		return nil, fmt.Errorf("external %s signer returned an invalid signature", signer.label)
	}
	return append([]byte(nil), response.Signature...), nil
}

func (signer *externalSigner) close() error {
	if signer == nil {
		return nil
	}
	zero(signer.publicKey)
	return signer.connection.Close()
}

func serveBroker(descriptor int, parsed brokerOptions) error {
	if err := hardenProcess(); err != nil {
		return fmt.Errorf("could not harden broker process: %w", err)
	}
	selfPath, err := os.Executable()
	if err != nil {
		return fmt.Errorf("could not resolve broker executable: %w", err)
	}
	if _, err := validateTrustedNativeExecutable(selfPath); err != nil {
		return fmt.Errorf("broker executable or identity is not protected: %w", err)
	}
	syscall.CloseOnExec(descriptor)
	connection := os.NewFile(uintptr(descriptor), "signing-broker")
	if connection == nil {
		return errors.New("broker descriptor is invalid")
	}
	defer connection.Close()

	var initialization brokerRequest
	if err := receiveFrame(connection, &initialization); err != nil {
		return err
	}
	if initialization.Version != protocolVersion ||
		initialization.ID != 0 ||
		initialization.Operation != "initialize" ||
		!initialization.hasValidInitializationShape() {
		zeroRequestSecrets(&initialization)
		_ = sendError(connection, 0, "Signing broker initialization is malformed")
		return errors.New("initialization is malformed")
	}
	var applySigner *externalSigner
	var evalSigner *externalSigner
	status := brokerStatus{
		Capabilities: make([]string, 0, 2),
		ApplyPublicKey: append(
			[]byte(nil), initialization.ApplyPublicKey...,
		),
		EvalPublicKey: append(
			[]byte(nil), initialization.EvalPublicKey...,
		),
		CorpusReleasePublicKey: append(
			[]byte(nil), initialization.CorpusReleasePublicKey...,
		),
		CorpusReleasePublicKeys: clonePublicKeys(
			initialization.CorpusReleasePublicKeys,
		),
	}
	if parsed.applySignerFD >= 0 {
		var err error
		applySigner, err = connectExternalSigner(
			"apply", parsed.applySignerFD, initialization.ApplyPublicKey,
		)
		if err != nil {
			zeroRequestSecrets(&initialization)
			_ = sendError(connection, 0, "External apply signer initialization failed")
			return err
		}
		defer applySigner.close() //nolint:errcheck
		status.Capabilities = append(status.Capabilities, "apply_ed25519")
	}
	if parsed.evalSignerFD >= 0 {
		var err error
		evalSigner, err = connectExternalSigner(
			"eval", parsed.evalSignerFD, initialization.EvalPublicKey,
		)
		if err != nil {
			zeroRequestSecrets(&initialization)
			_ = sendError(connection, 0, "External eval signer initialization failed")
			return err
		}
		defer evalSigner.close() //nolint:errcheck
		status.Capabilities = append(status.Capabilities, "eval_ed25519")
	}
	zeroRequestSecrets(&initialization)
	if err := sendFrame(connection, brokerResponse{
		Version: protocolVersion,
		ID:      0,
		OK:      true,
		Result:  status,
	}); err != nil {
		return err
	}
	if err := authenticateBrokerPeer(connection); err != nil {
		return fmt.Errorf("could not authenticate broker peer: %w", err)
	}

	lastRequestID := int64(0)
	for {
		var request brokerRequest
		if err := receiveFrame(connection, &request); err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				return nil
			}
			return err
		}
		if err := validateBrokerRequestEnvelope(&request, &lastRequestID); err != nil {
			zeroRequestSecrets(&request)
			if sendErr := sendError(connection, request.ID, err.Error()); sendErr != nil {
				return sendErr
			}
			continue
		}
		if err := validateBrokerRequestOperation(&request); err != nil {
			zeroRequestSecrets(&request)
			if sendErr := sendError(connection, request.ID, err.Error()); sendErr != nil {
				return sendErr
			}
			continue
		}
		switch request.Operation {
		case "shutdown":
			zero(request.Payload)
			return nil
		case "status":
			if err := sendFrame(connection, brokerResponse{
				Version: protocolVersion,
				ID:      request.ID,
				OK:      true,
				Result:  status,
			}); err != nil {
				return err
			}
		case "apply_ed25519_sign":
			if applySigner == nil {
				if err := sendError(connection, request.ID, "Apply manifest signer is not provisioned"); err != nil {
					return err
				}
				continue
			}
			signature, signErr := applySigner.sign(request.Payload)
			zero(request.Payload)
			if signErr != nil {
				_ = sendError(connection, request.ID, "External apply signer failed")
				return signErr
			}
			err := sendFrame(connection, brokerResponse{
				Version: protocolVersion,
				ID:      request.ID,
				OK:      true,
				Result:  map[string][]byte{"signature": signature},
			})
			zero(signature)
			if err != nil {
				return err
			}
		case "eval_ed25519_sign":
			if evalSigner == nil {
				if err := sendError(connection, request.ID, "Eval evidence signer is not provisioned"); err != nil {
					return err
				}
				continue
			}
			signature, signErr := evalSigner.sign(request.Payload)
			zero(request.Payload)
			if signErr != nil {
				_ = sendError(connection, request.ID, "External eval signer failed")
				return signErr
			}
			err := sendFrame(connection, brokerResponse{
				Version: protocolVersion,
				ID:      request.ID,
				OK:      true,
				Result:  map[string][]byte{"signature": signature},
			})
			zero(signature)
			if err != nil {
				return err
			}
		}
	}
}

func zeroRequestSecrets(request *brokerRequest) {
	zero(request.Payload)
	zero(request.ApplyPublicKey)
	zero(request.EvalPublicKey)
	zero(request.CorpusReleasePublicKey)
	zeroPublicKeys(request.CorpusReleasePublicKeys)
}

func sendError(connection io.Writer, requestID int64, message string) error {
	return sendFrame(connection, brokerResponse{
		Version: protocolVersion,
		ID:      requestID,
		OK:      false,
		Error:   message,
		Result:  map[string]any{},
	})
}

func sendFrame(connection io.Writer, payload any) error {
	raw, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	defer zero(raw)
	if len(raw) > maxFrameBytes {
		return errors.New("signing broker frame exceeds its size limit")
	}
	header := make([]byte, 4)
	binary.BigEndian.PutUint32(header, uint32(len(raw)))
	if err := writeAll(connection, header); err != nil {
		return err
	}
	return writeAll(connection, raw)
}

func receiveFrame(connection io.Reader, destination any) error {
	header := make([]byte, 4)
	if _, err := io.ReadFull(connection, header); err != nil {
		return err
	}
	length := binary.BigEndian.Uint32(header)
	if length > maxFrameBytes {
		return errors.New("signing broker frame exceeds its size limit")
	}
	raw := make([]byte, int(length))
	defer zero(raw)
	if _, err := io.ReadFull(connection, raw); err != nil {
		return err
	}
	if err := rejectDuplicateJSONKeys(raw); err != nil {
		return fmt.Errorf("signing broker frame is malformed: %w", err)
	}
	if _, sensitiveRequest := destination.(*brokerRequest); sensitiveRequest {
		// brokerRequest owns its strict unknown-field and field-presence decode.
		// Decode directly from the frame so json.Decoder cannot retain a second,
		// unzeroizable copy of signing payloads in its read buffer.
		if err := json.Unmarshal(raw, destination); err != nil {
			return fmt.Errorf("signing broker frame is malformed: %w", err)
		}
		return nil
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return fmt.Errorf("signing broker frame is malformed: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("signing broker frame contains trailing JSON")
		}
		return fmt.Errorf("signing broker frame has malformed trailing data: %w", err)
	}
	return nil
}

func rejectDuplicateJSONKeys(raw []byte) error {
	if !json.Valid(raw) {
		return errors.New("frame is not valid JSON or contains trailing JSON")
	}
	scanner := jsonKeyScanner{raw: raw}
	if err := scanner.scanValue(0); err != nil {
		return err
	}
	scanner.skipWhitespace()
	if scanner.offset != len(raw) {
		return errors.New("frame contains trailing JSON")
	}
	return nil
}

// jsonKeyScanner validates duplicate object keys without decoding JSON values.
// In particular, signing-payload strings are never materialized as immutable Go
// strings merely to inspect the request shape. json.Valid runs first, so this
// scanner only needs to walk an already-valid JSON grammar.
type jsonKeyScanner struct {
	raw    []byte
	offset int
}

func (scanner *jsonKeyScanner) skipWhitespace() {
	for scanner.offset < len(scanner.raw) {
		switch scanner.raw[scanner.offset] {
		case ' ', '\t', '\n', '\r':
			scanner.offset++
		default:
			return
		}
	}
}

func (scanner *jsonKeyScanner) scanValue(depth int) error {
	if depth > maxJSONDepth {
		return fmt.Errorf("JSON nesting exceeds the %d-level limit", maxJSONDepth)
	}
	scanner.skipWhitespace()
	if scanner.offset >= len(scanner.raw) {
		return io.ErrUnexpectedEOF
	}
	switch scanner.raw[scanner.offset] {
	case '{':
		return scanner.scanObject(depth)
	case '[':
		return scanner.scanArray(depth)
	case '"':
		_, err := scanner.scanString()
		return err
	default:
		start := scanner.offset
		for scanner.offset < len(scanner.raw) {
			switch scanner.raw[scanner.offset] {
			case ' ', '\t', '\n', '\r', ',', ']', '}':
				if scanner.offset == start {
					return errors.New("JSON value is malformed")
				}
				return nil
			default:
				scanner.offset++
			}
		}
		if scanner.offset == start {
			return errors.New("JSON value is malformed")
		}
		return nil
	}
}

func (scanner *jsonKeyScanner) scanObject(depth int) error {
	scanner.offset++
	scanner.skipWhitespace()
	if scanner.raw[scanner.offset] == '}' {
		scanner.offset++
		return nil
	}
	seen := make(map[string]struct{})
	for {
		scanner.skipWhitespace()
		if scanner.offset >= len(scanner.raw) || scanner.raw[scanner.offset] != '"' {
			return errors.New("object key is not a string")
		}
		encodedKey, err := scanner.scanString()
		if err != nil {
			return err
		}
		if len(encodedKey) > maxJSONKeyBytes {
			return fmt.Errorf("object key exceeds %d bytes", maxJSONKeyBytes)
		}
		var key string
		if err := json.Unmarshal(encodedKey, &key); err != nil {
			return err
		}
		if _, duplicate := seen[key]; duplicate {
			return fmt.Errorf("duplicate object key %q", key)
		}
		seen[key] = struct{}{}
		scanner.skipWhitespace()
		if scanner.raw[scanner.offset] != ':' {
			return errors.New("object key is missing its value separator")
		}
		scanner.offset++
		if err := scanner.scanValue(depth + 1); err != nil {
			return err
		}
		scanner.skipWhitespace()
		switch scanner.raw[scanner.offset] {
		case '}':
			scanner.offset++
			return nil
		case ',':
			scanner.offset++
		default:
			return errors.New("object is not terminated")
		}
	}
}

func (scanner *jsonKeyScanner) scanArray(depth int) error {
	scanner.offset++
	scanner.skipWhitespace()
	if scanner.raw[scanner.offset] == ']' {
		scanner.offset++
		return nil
	}
	for {
		if err := scanner.scanValue(depth + 1); err != nil {
			return err
		}
		scanner.skipWhitespace()
		switch scanner.raw[scanner.offset] {
		case ']':
			scanner.offset++
			return nil
		case ',':
			scanner.offset++
		default:
			return errors.New("array is not terminated")
		}
	}
}

func (scanner *jsonKeyScanner) scanString() ([]byte, error) {
	start := scanner.offset
	scanner.offset++
	for scanner.offset < len(scanner.raw) {
		switch scanner.raw[scanner.offset] {
		case '"':
			scanner.offset++
			return scanner.raw[start:scanner.offset], nil
		case '\\':
			scanner.offset += 2
		default:
			scanner.offset++
		}
	}
	return nil, errors.New("JSON string is not terminated")
}

func writeAll(writer io.Writer, data []byte) error {
	for len(data) != 0 {
		written, err := writer.Write(data)
		if err != nil {
			return err
		}
		if written == 0 {
			return io.ErrShortWrite
		}
		data = data[written:]
	}
	return nil
}

func equalStrings(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func equalByteSlices(left, right [][]byte) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if !bytes.Equal(left[index], right[index]) {
			return false
		}
	}
	return true
}

func clonePublicKeys(publicKeys [][]byte) [][]byte {
	cloned := make([][]byte, len(publicKeys))
	for index, publicKey := range publicKeys {
		cloned[index] = append([]byte(nil), publicKey...)
	}
	return cloned
}

func zero(value []byte) {
	for index := range value {
		value[index] = 0
	}
	runtime.KeepAlive(value)
}
