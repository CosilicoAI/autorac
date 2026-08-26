//go:build darwin || linux

package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"golang.org/x/sys/unix"
)

func encodedTestPublicKey(value byte) string {
	return base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{value}, 32))
}

func parseTestTrustRoots(t *testing.T, payload map[string]any) ([]byte, []byte, [][]byte, error) {
	t.Helper()
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	return parseProtectedTrustRoots(raw, "/protected/signing-trust-roots.json")
}

func baseTrustRoots() map[string]any {
	return map[string]any{
		"schema":                            "axiom-encode/signing-trust-roots/v2",
		"apply_ed25519_public_key":          encodedTestPublicKey(0xab),
		"eval_ed25519_public_key":           encodedTestPublicKey(0xcd),
		"corpus_release_ed25519_public_key": encodedTestPublicKey(0x17),
	}
}

func TestLoadProtectedTrustRootsAcceptsV2AndV3Keyring(t *testing.T) {
	v2 := baseTrustRoots()
	apply, eval, corpusKeys, err := parseTestTrustRoots(t, v2)
	if err != nil {
		t.Fatalf("v2 trust roots were rejected: %v", err)
	}
	if len(apply) != 32 || len(eval) != 32 || len(corpusKeys) != 1 || len(corpusKeys[0]) != 32 {
		t.Fatalf("v2 trust roots returned an invalid keyring: %#v", corpusKeys)
	}
	v2CorpusKey := append([]byte(nil), corpusKeys[0]...)

	v3Single := baseTrustRoots()
	v3Single["schema"] = "axiom-encode/signing-trust-roots/v3"
	delete(v3Single, "corpus_release_ed25519_public_key")
	v3Single["corpus_release_ed25519_public_keys"] = []string{
		encodedTestPublicKey(0x17),
	}
	_, _, corpusKeys, err = parseTestTrustRoots(t, v3Single)
	if err != nil {
		t.Fatalf("single-key v3 trust roots were rejected: %v", err)
	}
	if len(corpusKeys) != 1 || !bytes.Equal(corpusKeys[0], v2CorpusKey) {
		t.Fatalf("single-key v3 was not equivalent to v2: %#v", corpusKeys)
	}

	v3Consistent := baseTrustRoots()
	v3Consistent["schema"] = "axiom-encode/signing-trust-roots/v3"
	v3Consistent["corpus_release_ed25519_public_keys"] = []string{
		encodedTestPublicKey(0x17),
	}
	if _, _, _, err = parseTestTrustRoots(t, v3Consistent); err != nil {
		t.Fatalf("consistent singular/plural v3 trust roots were rejected: %v", err)
	}

	v3 := baseTrustRoots()
	v3["schema"] = "axiom-encode/signing-trust-roots/v3"
	delete(v3, "corpus_release_ed25519_public_key")
	v3["corpus_release_ed25519_public_keys"] = []string{
		encodedTestPublicKey(0x18),
		encodedTestPublicKey(0x17),
	}
	_, _, corpusKeys, err = parseTestTrustRoots(t, v3)
	if err != nil {
		t.Fatalf("v3 trust roots were rejected: %v", err)
	}
	if len(corpusKeys) != 2 || corpusKeys[0][0] != 0x18 || corpusKeys[1][0] != 0x17 {
		t.Fatalf("v3 trust roots lost keyring order: %#v", corpusKeys)
	}
}

func TestLoadProtectedTrustRootsRejectsInvalidV3Keyrings(t *testing.T) {
	for _, testCase := range []struct {
		name   string
		mutate func(map[string]any)
		error  string
	}{
		{
			name: "empty",
			mutate: func(payload map[string]any) {
				payload["corpus_release_ed25519_public_keys"] = []string{}
			},
			error: "must not be empty",
		},
		{
			name: "malformed base64",
			mutate: func(payload map[string]any) {
				payload["corpus_release_ed25519_public_keys"] = []string{"not-base64!!"}
			},
			error: "base64-encoded raw bytes",
		},
		{
			name: "wrong length",
			mutate: func(payload map[string]any) {
				payload["corpus_release_ed25519_public_keys"] = []string{base64.StdEncoding.EncodeToString([]byte("short"))}
			},
			error: "must contain 32 raw bytes",
		},
		{
			name: "conflicting singular and plural",
			mutate: func(payload map[string]any) {
				payload["corpus_release_ed25519_public_key"] = encodedTestPublicKey(0x19)
			},
			error: "singular and plural corpus release public keys conflict",
		},
		{
			name: "unknown schema",
			mutate: func(payload map[string]any) {
				payload["schema"] = "axiom-encode/signing-trust-roots/v999"
			},
			error: "schema is unsupported",
		},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			payload := baseTrustRoots()
			payload["schema"] = "axiom-encode/signing-trust-roots/v3"
			payload["corpus_release_ed25519_public_keys"] = []string{encodedTestPublicKey(0x18)}
			delete(payload, "corpus_release_ed25519_public_key")
			testCase.mutate(payload)
			_, _, _, err := parseTestTrustRoots(t, payload)
			if err == nil || !strings.Contains(err.Error(), testCase.error) {
				t.Fatalf("expected %q rejection, got %v", testCase.error, err)
			}
		})
	}
}

func TestCredentialOutboxDestinationRejectsDevice(t *testing.T) {
	fd, err := unix.Open("/dev", unix.O_RDONLY|unix.O_DIRECTORY|unix.O_CLOEXEC, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer unix.Close(fd)
	if err := validateCredentialOutboxDestination(fd, "null"); err == nil || !strings.Contains(err.Error(), "special file") {
		t.Fatalf("device destination was not rejected: %v", err)
	}
}

func TestCleanChildEnvironmentForwardsApplyCheckoutIdentity(t *testing.T) {
	values := map[string]string{
		"AXIOM_ENCODE_APPLY_CHECKOUT": "/home/runner/work/repo/repo/axiom-encode",
		"GITHUB_ACTIONS":              "true",
		"GITHUB_SHA":                  strings.Repeat("a", 40),
		"GITHUB_WORKSPACE":            "/home/runner/work/repo/repo",
		"UNTRUSTED_AMBIENT_VALUE":     "must-not-cross",
	}
	for name, value := range values {
		t.Setenv(name, value)
	}
	environment := cleanChildEnvironment(10, 11, "/trusted/bin", "/trusted")
	joined := "\n" + strings.Join(environment, "\n") + "\n"
	for name, value := range values {
		entry := "\n" + name + "=" + value + "\n"
		if name == "UNTRUSTED_AMBIENT_VALUE" {
			if strings.Contains(joined, entry) {
				t.Fatalf("untrusted ambient value crossed supervisor boundary")
			}
			continue
		}
		if !strings.Contains(joined, entry) {
			t.Fatalf("missing apply identity environment %s", name)
		}
	}
	if _, present := os.LookupEnv("AXIOM_ENCODE_APPLY_SIGNING_KEY"); present {
		t.Fatal("test environment unexpectedly contains an apply signing key")
	}
}

func framed(raw string) []byte {
	header := make([]byte, 4)
	binary.BigEndian.PutUint32(header, uint32(len(raw)))
	return append(header, []byte(raw)...)
}

func TestReceiveFrameRejectsTrailingJSONValue(t *testing.T) {
	var request brokerRequest
	err := receiveFrame(
		bytes.NewReader(framed(`{"version":4,"id":1,"op":"status"} {}`)),
		&request,
	)
	if err == nil || !strings.Contains(err.Error(), "trailing JSON") {
		t.Fatalf("expected trailing JSON rejection, got %v", err)
	}
}

func TestReceiveFrameRejectsUnknownField(t *testing.T) {
	var request brokerRequest
	err := receiveFrame(
		bytes.NewReader(framed(`{"version":4,"id":1,"op":"status","legacy_key":"no"}`)),
		&request,
	)
	if err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("expected unknown field rejection, got %v", err)
	}
}

func TestReceiveFrameRejectsDuplicateKeysAtEveryDepth(t *testing.T) {
	for _, raw := range []string{
		`{"version":4,"id":1,"id":2,"op":"status"}`,
		`{"version":4,"\u0069d":1,"id":2,"op":"status"}`,
		`{"version":4,"id":1,"op":"status","payload":{"x":1,"x":2}}`,
	} {
		var request brokerRequest
		err := receiveFrame(bytes.NewReader(framed(raw)), &request)
		if err == nil || !strings.Contains(err.Error(), "duplicate object key") {
			t.Fatalf("expected duplicate-key rejection for %s, got %v", raw, err)
		}
	}
}

func TestReceiveFrameRejectsOversizedObjectKey(t *testing.T) {
	raw := `{"` + strings.Repeat("x", maxJSONKeyBytes+1) + `":1}`
	var request brokerRequest
	err := receiveFrame(bytes.NewReader(framed(raw)), &request)
	if err == nil || !strings.Contains(err.Error(), "object key exceeds") {
		t.Fatalf("expected oversized-key rejection, got %v", err)
	}
}

func TestReceiveFrameRejectsExcessiveJSONNesting(t *testing.T) {
	raw := `{"version":4,"id":1,"op":"status","payload":` +
		strings.Repeat("[", maxJSONDepth+2) + `null` +
		strings.Repeat("]", maxJSONDepth+2) + `}`
	var request brokerRequest
	err := receiveFrame(bytes.NewReader(framed(raw)), &request)
	if err == nil || !strings.Contains(err.Error(), "nesting exceeds") {
		t.Fatalf("expected excessive-nesting rejection, got %v", err)
	}
}

func decodedRequest(t *testing.T, raw string) brokerRequest {
	t.Helper()
	var request brokerRequest
	if err := receiveFrame(bytes.NewReader(framed(raw)), &request); err != nil {
		t.Fatalf("could not decode request: %v", err)
	}
	return request
}

func TestBrokerRequestRequiresExactOperationFields(t *testing.T) {
	testCases := []struct {
		name  string
		raw   string
		error string
	}{
		{
			name:  "empty payload on status",
			raw:   `{"version":4,"id":1,"op":"status","payload":""}`,
			error: "status request is malformed",
		},
		{
			name:  "null payload on sign",
			raw:   `{"version":4,"id":1,"op":"apply_ed25519_sign","payload":null}`,
			error: "signing request is malformed",
		},
		{
			name:  "missing payload on sign",
			raw:   `{"version":4,"id":1,"op":"apply_ed25519_sign"}`,
			error: "signing request is malformed",
		},
		{
			name:  "empty forbidden initialization field",
			raw:   `{"version":4,"id":1,"op":"status","apply_public_key":""}`,
			error: "forbidden initialization fields",
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			request := decodedRequest(t, testCase.raw)
			err := validateBrokerRequestOperation(&request)
			if err == nil || !strings.Contains(err.Error(), testCase.error) {
				t.Fatalf("expected %q rejection, got %v", testCase.error, err)
			}
		})
	}

	request := decodedRequest(
		t,
		`{"version":4,"id":1,"op":"apply_ed25519_sign","payload":""}`,
	)
	if err := validateBrokerRequestOperation(&request); err != nil {
		t.Fatalf("explicit empty signing payload should be valid: %v", err)
	}
}

func TestBrokerRequestIDsArePositiveAndStrictlyIncreasing(t *testing.T) {
	lastRequestID := int64(0)
	first := decodedRequest(t, `{"version":4,"id":1,"op":"status"}`)
	if err := validateBrokerRequestEnvelope(&first, &lastRequestID); err != nil {
		t.Fatalf("first request should be valid: %v", err)
	}
	for _, raw := range []string{
		`{"version":4,"id":1,"op":"status"}`,
		`{"version":4,"id":0,"op":"status"}`,
		`{"version":4,"id":-1,"op":"status"}`,
	} {
		request := decodedRequest(t, raw)
		err := validateBrokerRequestEnvelope(&request, &lastRequestID)
		if err == nil || !strings.Contains(err.Error(), "strictly increasing") {
			t.Fatalf("expected request ID rejection for %s, got %v", raw, err)
		}
	}
	third := decodedRequest(t, `{"version":4,"id":3,"op":"status"}`)
	if err := validateBrokerRequestEnvelope(&third, &lastRequestID); err != nil {
		t.Fatalf("later increasing request should be valid: %v", err)
	}
}

func TestBrokerRejectsRemovedV3Protocol(t *testing.T) {
	request := decodedRequest(t, `{"version":3,"id":1,"op":"status"}`)
	lastRequestID := int64(0)
	err := validateBrokerRequestEnvelope(&request, &lastRequestID)
	if err == nil || !strings.Contains(err.Error(), "malformed") {
		t.Fatalf("expected removed protocol rejection, got %v", err)
	}
}

func TestBrokerInitializationRejectsEmptyOrExtraFields(t *testing.T) {
	for _, raw := range []string{
		`{"version":4,"id":0,"op":"initialize","apply_public_key":""}`,
		`{"version":4,"id":0,"op":"initialize","apply_public_key":"q6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6s=","payload":""}`,
		`{"version":4,"id":0,"op":"initialize"}`,
	} {
		request := decodedRequest(t, raw)
		if request.hasValidInitializationShape() {
			t.Fatalf("expected initialization shape rejection for %s", raw)
		}
	}
}

func TestBrokerInitializationRequiresThreeDistinctRoots(t *testing.T) {
	valid := decodedRequest(
		t,
		`{"version":4,"id":0,"op":"initialize","apply_public_key":"q6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6s=","eval_public_key":"zc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc0=","corpus_release_public_key":"FxcXFxcXFxcXFxcXFxcXFxcXFxcXFxcXFxcXFxcXFxc=","corpus_release_public_keys":["FxcXFxcXFxcXFxcXFxcXFxcXFxcXFxcXFxcXFxcXFxc="]}`,
	)
	if !valid.hasValidInitializationShape() {
		t.Fatal("expected three distinct initialization roots to be valid")
	}

	aliased := valid
	aliased.CorpusReleasePublicKey = append([]byte(nil), valid.ApplyPublicKey...)
	if aliased.hasValidInitializationShape() {
		t.Fatal("expected aliased corpus release root to be rejected")
	}

	conflicting := valid
	conflicting.CorpusReleasePublicKeys = [][]byte{bytes.Repeat([]byte{0x18}, 32)}
	if conflicting.hasValidInitializationShape() {
		t.Fatal("expected conflicting corpus release keyring to be rejected")
	}
}

func TestParseRawPublicKey(t *testing.T) {
	publicKey := bytes.Repeat([]byte{0xab}, 32)
	encoded := []byte("q6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6s=")
	parsed, err := parsePublicKey(encoded)
	if err != nil {
		t.Fatal(err)
	}
	defer zero(parsed)
	if !bytes.Equal(parsed, publicKey) {
		t.Fatal("parsed public key did not match")
	}
}

func TestBrokerAllowsVerificationOnlyButRejectsAliasedSignerDescriptors(t *testing.T) {
	parsed, err := parseBrokerOptions(nil)
	if err != nil || parsed.applySignerFD != -1 || parsed.evalSignerFD != -1 {
		t.Fatalf("expected verification-only broker options, got %#v err=%v", parsed, err)
	}
	_, err = parseBrokerOptions([]string{
		"--apply-signer-fd", "4", "--eval-signer-fd", "4",
	})
	if err == nil || !strings.Contains(err.Error(), "must be distinct") {
		t.Fatalf("expected aliased signer descriptor rejection, got %v", err)
	}
}

func TestCleanChildEnvironmentSetsTrustedRuntimeMarker(t *testing.T) {
	environment := cleanChildEnvironment(1234, 3, "/opt/axiom-verification", "/root")
	want := map[string]string{
		trustedRuntimeEnv:  "1",
		brokerActiveEnv:    "1",
		"PYTHONUNBUFFERED": "1",
	}
	for name, value := range want {
		expected := name + "=" + value
		found := false
		for _, entry := range environment {
			if entry == expected {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("cleaned child environment missing %q; got %v", expected, environment)
		}
	}
	// The marker must originate from the empty child environment, never inherited.
	for _, entry := range environment {
		if strings.HasPrefix(entry, trustedRuntimeEnv+"=") && entry != trustedRuntimeEnv+"=1" {
			t.Fatalf("trusted-runtime marker must be exactly 1, got %q", entry)
		}
	}
}

func TestCodexScratchPolicyRejectsRootOwnedHomeForNonRootRuntime(t *testing.T) {
	if err := validateCodexScratchPolicy(os.ModeDir|0700, 0, 1000); err == nil ||
		!strings.Contains(err.Error(), "not runtime-owned") {
		t.Fatalf("expected root-owned scratch rejection, got %v", err)
	}
}

func TestCodexScratchPolicyAcceptsOperatorOwnedProtectedHome(t *testing.T) {
	if err := validateCodexScratchPolicy(os.ModeDir|0700, 1000, 1000); err != nil {
		t.Fatalf("expected operator-owned 0700 scratch acceptance, got %v", err)
	}
}
