# Strategist Reconciliation (T9.0)

This document is the normative reconciliation reference for the Phase 9 Strategist overhaul (`wow_bot.strategist`). It audits all pre-lab strategist symbols, classifies their relationship to Phase 9 tasks (T9.1–T9.4), establishes module naming conventions, and defines frozen interfaces for Phase 9 development.

---

## Inventory

Below is the complete inventory of public top-level symbols defined in `src/wow_bot/strategist/`.

| Name | Kind | Module | Description | Exported via `__init__.py` | Exercised in `tests/` |
|---|---|---|---|---|---|
| `LLMClient` | class | `src/wow_bot/strategist/llm_client.py` | Async client wrapper around AsyncOpenAI for local Ollama chat completions with endpoint validation and retries. | No | Yes (`tests/unit/test_llm_client.py`, `tests/unit/test_strategist.py`, `tests/unit/test_strategy_parser.py`, `tests/unit/test_review_gate.py`) |
| `LLMResponseError` | exception | `src/wow_bot/strategist/llm_client.py` | Exception raised when the local LLM endpoint returns empty choices or None completion content. | No | Yes (`tests/unit/test_strategist.py`) |
| `LOCAL_HOSTNAMES` | constant | `src/wow_bot/strategist/llm_client.py` | Frozen set of allowed local loopback hostnames/IPs (`127.0.0.1`, `localhost`, `::1`, `[::1]`) for endpoint safety checks. | No | No (exercised implicitly by `_validate_local_endpoint` unit tests in `tests/unit/test_llm_client.py`) |
| `DEFAULT_STRATEGY_TTL_SECONDS` | constant | `src/wow_bot/strategist/orchestrator.py` | Default fallback strategy duration constant in seconds (1800.0s / 30 minutes). | No | No (exercised implicitly by strategy validity unit tests in `tests/unit/test_strategist.py`) |
| `Strategist` | class | `src/wow_bot/strategist/orchestrator.py` | Pre-lab strategy orchestrator class composing prompt builder, local LLM client, and strategy response parser. | No | Yes (`tests/unit/test_strategist.py`, `tests/unit/test_main.py`, `tests/unit/test_prompts.py`, `tests/test_world_summary.py`) |
| `EXPECTED_TOP_LEVEL_KEYS` | constant | `src/wow_bot/strategist/parser.py` | Frozen set of top-level JSON keys required by pre-lab strategy schema validation. | No | No (exercised implicitly by strategy parsing unit tests in `tests/unit/test_strategy_parser.py`) |
| `EXPECTED_CONSTRAINT_KEYS` | constant | `src/wow_bot/strategist/parser.py` | Frozen set of constraint object keys required by pre-lab strategy schema validation. | No | No (exercised implicitly by strategy constraint unit tests in `tests/unit/test_strategy_parser.py`) |
| `StrategyParseError` | exception | `src/wow_bot/strategist/parser.py` | Exception raised when raw LLM output violates strategy JSON formatting or schema validation contracts. | No | Yes (`tests/unit/test_strategy_parser.py`, `tests/unit/test_review_gate.py`, `tests/unit/test_strategist.py`) |
| `parse_strategy_response` | function | `src/wow_bot/strategist/parser.py` | Pure function parsing raw LLM JSON text output into a validated domain Strategy object. | No | Yes (`tests/unit/test_strategy_parser.py`, `tests/unit/test_review_gate.py`) |
| `DynamicContext` | class | `src/wow_bot/strategist/prompts.py` | Frozen dataclass encapsulating runtime context (current time, session start, available regions, previous strategy). | Yes (`# DEPRECATED: see STRATEGIST_RECONCILIATION.md`) | Yes (`tests/unit/test_prompts.py`, `tests/unit/test_strategist.py`) |
| `load_system_prompt` | function | `src/wow_bot/strategist/prompts.py` | Cached function loading and returning persistent persona system prompt text from system_prompt.txt. | Yes | Yes (`tests/unit/test_prompts.py`) |
| `build_user_prompt` | function | `src/wow_bot/strategist/prompts.py` | Pure function rendering user prompt string from MetaState and DynamicContext inputs. | Yes (`# DEPRECATED: see STRATEGIST_RECONCILIATION.md`) | Yes (`tests/unit/test_prompts.py`) |

---

## Classification

Each public symbol is classified into exactly one of `{KEEP_AS_IS, KEEP_AND_EXTEND, RENAME, DEPRECATE, REPLACE}`:

1. `LLMClient`: `KEEP_AS_IS` — T9.4 (`orchestrator_v2.py`) will use `LLMClient` directly for local Ollama chat completions without API modifications.
2. `LLMResponseError`: `KEEP_AS_IS` — Core error exception raised by `LLMClient` when completions return empty choices or content, required by T9.4 error handling.
3. `LOCAL_HOSTNAMES`: `KEEP_AS_IS` — Frozen set of allowed local loopback hostnames used by `LLMClient` safety validation.
4. `DEFAULT_STRATEGY_TTL_SECONDS`: `KEEP_AS_IS` — Standard strategy expiration TTL constant reusable across pre-lab and Phase 9 orchestrator implementations.
5. `Strategist`: `DEPRECATE` — Pre-lab strategy orchestrator is superseded by T9.4 `StrategistV2` in `orchestrator_v2.py`.
6. `EXPECTED_TOP_LEVEL_KEYS`: `DEPRECATE` — Pre-lab JSON schema constants are superseded by T9.1 grounded prompt schema and T9.3 goal vocabulary.
7. `EXPECTED_CONSTRAINT_KEYS`: `DEPRECATE` — Pre-lab constraint keys schema is superseded by T9.1 grounded prompt schema and T9.3 goal vocabulary.
8. `StrategyParseError`: `KEEP_AS_IS` — Core exception class raised when raw LLM response string fails JSON decoding or schema validation, reused by T9.3/T9.4 parsing.
9. `parse_strategy_response`: `DEPRECATE` — Pre-lab response parser is superseded by T9.3 vocabulary guard (`vocab.py`) and T9.4 response parsing in Phase 9.
10. `DynamicContext`: `DEPRECATE` — Pre-lab prompt context structure is superseded by T9.1 grounded prompt builder (`prompts_v2.py`) input parameters.
11. `load_system_prompt`: `KEEP_AS_IS` — System persona prompt loader function continues to be used by grounded prompt building in T9.1.
12. `build_user_prompt`: `DEPRECATE` — Pre-lab prompt builder function is superseded by T9.1 grounded prompt builder (`prompts_v2.py`) consuming `MetaState`, `WorldSummary`, and `GameState`.

*Note on RENAME and REPLACE*: No symbol was classified as `RENAME` or `REPLACE`. All pre-lab symbols are either reused directly (`KEEP_AS_IS`) or retained with deprecation notices (`DEPRECATE`) to guarantee backwards compatibility for existing tests under Strategy A.

---

## Naming Convention

**Chosen Strategy: STRATEGY A (versioned modules)**

Phase 9 implements versioned and dedicated module names alongside existing pre-lab modules:
- Pre-lab modules retain their file names (`prompts.py`, `orchestrator.py`, `parser.py`, `llm_client.py`).
- Phase 9 capabilities are added in dedicated, versioned modules:
  - `prompts_v2.py` (T9.1 Grounded prompt builder)
  - `cooldown.py` (T9.2 Dynamic cooldown gate)
  - `vocab.py` (T9.3 Vocabulary guard)
  - `orchestrator_v2.py` (T9.4 Strategist Orchestrator v2)

**Rationale:** Strategy A eliminates all risk of breaking pre-lab tests that import directly from pre-lab strategist modules. It provides a clean, side-by-side transition during Phase 9 development while ensuring pre-lab MOCK_MODE tests remain 100% passing and reproducible.

---

## Frozen Interface

The following contract specifies the strict interface boundaries for authors implementing Phase 9 tasks T9.1 through T9.4.

### 1. `src/wow_bot/strategist/llm_client.py` (T9.4 integration)
- **MUST NOT CHANGE:** `LLMClient`, `LLMResponseError`, `LOCAL_HOSTNAMES`, `LLMClient.__init__`, `LLMClient.query`, `LLMClient.health_check`, `LLMClient.close`.
- **MAY ADD:** Additional properties or helper methods for response latency or prompt hash tracking if required by T9.4.
- **MAY REFACTOR FREELY:** Private functions `_validate_local_endpoint` and `_is_transient_error`.

### 2. `src/wow_bot/strategist/prompts.py` (Coexists with T9.1 `prompts_v2.py`)
- **MUST NOT CHANGE:** `load_system_prompt`, `build_user_prompt`, `DynamicContext`.
- **MAY ADD:** None. Pre-lab prompt module is frozen; T9.1 prompt builder belongs in `prompts_v2.py`.
- **MAY REFACTOR FREELY:** Private formatting helpers `_get_day_part`, `_estimate_session_remaining_minutes`, `_format_time_context`, `_format_internal_state`, `_format_events`, `_format_previous_strategy`, `_format_available_regions`.

### 3. `src/wow_bot/strategist/parser.py` (Coexists with T9.3 `vocab.py` and T9.4)
- **MUST NOT CHANGE:** `StrategyParseError`, `parse_strategy_response`, `EXPECTED_TOP_LEVEL_KEYS`, `EXPECTED_CONSTRAINT_KEYS`.
- **MAY ADD:** None. Pre-lab parser module is frozen; Phase 9 vocabulary validation belongs in `vocab.py`.
- **MAY REFACTOR FREELY:** Private helpers `_reject_non_finite_constant`, `_unique_object`, `_unwrap_optional_code_fence`.

### 4. `src/wow_bot/strategist/orchestrator.py` (Coexists with T9.4 `orchestrator_v2.py`)
- **MUST NOT CHANGE:** `Strategist`, `DEFAULT_STRATEGY_TTL_SECONDS`, `Strategist.current_strategy`, `Strategist.is_expired`, `Strategist.generate_strategy`.
- **MAY ADD:** None. Pre-lab orchestrator module is frozen; T9.4 orchestrator belongs in `orchestrator_v2.py`.
- **MAY REFACTOR FREELY:** Private functions `_validate_timestamp` and `_EXPECTED_GENERATION_ERRORS`.

### 5. `src/wow_bot/strategist/__init__.py`
- **MUST NOT CHANGE:** Re-exported public symbols and backward-compatibility exports for pre-lab tests (`load_system_prompt`, `DynamicContext`, `build_user_prompt`).
- **MAY ADD:** Re-exports for Phase 9 symbols (`prompts_v2`, `orchestrator_v2`, `cooldown`, `vocab`, `StrategistV2`) as T9.1–T9.4 are completed.
