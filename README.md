# TRIBOT

TRIBOT is a research pipeline for evidence-locked Emergency Severity Index (ESI) reasoning over multi-turn triage dialogue. It compiles raw dialogue into traceable clinical evidence, qualifies ESI Step A/B policy candidates, isolates valid vital-sign facts, learns an interpretable ordinal resource model, and audits errors before deterministic policy reconciliation.

> **Research use only.** This repository is not a medical device and must not be used for autonomous triage, diagnosis, or treatment. The current artifacts are development-stage results on deidentified or synthetic dialogue and are not evidence of clinical safety or deployment readiness.

## Design Principles

- **Evidence locked:** every clinical claim and model feature points back to an evidence or measurement ID.
- **Prediction safe:** labels, personas, historical triage traces, and quarantined measurements are excluded from stages 01-04 prediction inputs.
- **Policy constrained:** LLMs qualify bounded candidates; they cannot invent evidence or directly override ESI policy.
- **Case grouped:** multiple dialogue realizations of one case remain in the same split or fold.
- **Fail closed:** unresolved conflicts and low-confidence resource predictions enter review rather than receiving a forced label.
- **Auditable:** each stage emits validation gates and SHA-256 lineage manifests.

## Pipeline

```text
Raw transcript JSON
        |
        v
00  Dataset contract, identity, leakage and measurement audit
        |
        v
01  Dialogue-to-evidence compiler
    deterministic concept spans + bounded Qwen resolution + NLI verification
        |
        +----------------------+
        v                      v
02  Policy trajectories       03  Vital-state facts
    + bounded DeepSeek             + ESI Decision Point D flags
        |                      |
        +----------+-----------+
                   v
04  Evidence-linked conditional ordinal resource learning (0 / 1 / 2+)
                   |
                   v
05  DeepSeek error-attribution harness (audit only)
05b Gold-label observability audit (audit only)
                   |
                   v
06  Deterministic policy reconciliation, selective review and loop actions
```

### Stage Responsibilities

| Stage | Responsibility | Must not do |
|---|---|---|
| 00 | Validate schema, IDs, duplicate groups, splits, label consistency and measurement plausibility | Supply audit labels or statistics to prediction stages |
| 01 | Produce local evidence cards with span, source, subject, assertion, temporality, episode relation and provenance | Assign ESI or policy outcomes |
| 02 | Reduce same-atom evidence into trajectories, map atoms to ESI policy fields, and qualify bounded Step A/B candidates | Consume invalid measurements or let an LLM invent evidence |
| 03 | Deduplicate canonical measurements and dialogue reveals; evaluate official vital policy signals | Convert a danger-zone flag directly into ESI 2 or a resource count |
| 04 | Build a physically separate feature plane and supervision plane; fit an interpretable conditional ordinal model | Use Step A/B, acuity, triage, or official danger-zone flags as model features |
| 05/05b | Attribute pipeline errors and assess whether gold ESI is observable from supplied dialogue | Rewrite labels or predictions automatically |
| 06 | Apply deterministic ESI policy ordering and abstention/review rules | Train a model or permit LLM audit results to override policy |

## Repository Layout

```text
src/
  00_audit_dataset.py
  01_build_evidence_cards.py
  01_score_nli.py
  02_build_qualified_trajectories.py
  02_qualify_with_deepseek.py
  03_build_vital_state_facts.py
  04_build_resource_learning_plane.py
  04_train_fused_conditional.py
  04_build_weak_resource_concept_probe.py
  04_probe_nonlinear_ceiling.py
  05_deepseek_error_harness.py
  05b_gold_esi_observability_harness.py
  06_reconcile_and_loop_v2.py
  clinical_concept_registry.py
  clinical_data_contract.py
  esi_policy_manifest_v5.py
  esi_vital_policy_manifest_v1.py
transcripts/data/   # local input data; intentionally not tracked
outputs/            # generated artifacts; intentionally not tracked
```

`project/`, `research/`, and `06_reconcile_and_loop.py` are legacy development material. They are not part of the current 00-06 execution chain. The current reconciliation entry point is `06_reconcile_and_loop_v2.py`.

## Input Contract

The current release expects one JSON file per dialogue realization under `transcripts/data/`. Required identity fields are:

- `case_id`: groups realizations of the same underlying case.
- `run_uuid`: identifies one dialogue realization.
- `history`: ordered patient, nurse, and system events.
- `vignette`: prediction-safe structured measurements and chief complaint.
- `ground_truth`: audit/supervision only; never consumed by stages 01-03 prediction logic.

The checked data snapshot contains 1,010 realizations from 541 cases. Several gates in stages 00-06 intentionally encode these snapshot counts. Change the corresponding expected-count arguments or gates before using another dataset.

The shared measurement contract currently assumes Fahrenheit temperature and quarantines pain outside 0-10 and implausible vital values. Review `src/clinical_data_contract.py` before using data with different units or populations.

## Installation

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Stage 01 also requires:

1. An OpenAI-compatible local Qwen endpoint (the checked configuration uses `Qwen2.5-7B-Instruct-AWQ` at `http://127.0.0.1:9097/v1/chat/completions`).
2. A local three-way NLI sequence-classification model compatible with Hugging Face Transformers.

Copy `src/.env.example` to `src/.env` and provide the API key locally. Never commit `.env` files.

## End-to-End Run

Run commands from the repository root.

### 00. Audit the dataset

```powershell
python src/00_audit_dataset.py
```

The script replaces its selected output directory. Do not point `--output-dir` at a directory containing unrelated files.

### 01. Compile evidence cards

Start the local Qwen endpoint, then prepare deterministic evidence and bounded ambiguity requests:

```powershell
python src/01_build_evidence_cards.py --stage prepare
```

Score the generated controlled hypotheses with a local NLI model:

```powershell
python src/01_score_nli.py `
  --input outputs/01_evidence_cards_v1/nli_pairs.jsonl `
  --output outputs/01_evidence_cards_v1/nli_scores.jsonl `
  --model <LOCAL_NLI_MODEL_PATH> `
  --device cuda
```

Finalize only after all NLI pairs have scores:

```powershell
python src/01_build_evidence_cards.py --stage finalize
```

### 02. Build and qualify policy trajectories

First run the deterministic dry stage:

```powershell
python src/02_build_qualified_trajectories.py `
  --input outputs/01_evidence_cards_v1/evidence_timelines.jsonl `
  --output-dir outputs/02_policy_trajectories_v1_dry_full
```

Then qualify the bounded candidates with DeepSeek:

```powershell
python src/02_qualify_with_deepseek.py `
  --dry-output-dir outputs/02_policy_trajectories_v1_dry_full `
  --output-dir outputs/02_policy_trajectories_v1_deepseek_full `
  --env-file src/.env
```

### 03. Build vital-state facts

```powershell
python src/03_build_vital_state_facts.py
```

### 04. Build the learning plane and train the ordinal model

```powershell
python src/04_build_resource_learning_plane.py `
  --transcripts transcripts/data `
  --stage02 outputs/02_policy_trajectories_v1_deepseek_full/02_qualified_trajectories.jsonl `
  --stage03 outputs/03_vital_state_facts_v1/03_vital_state_facts.jsonl `
  --output-dir outputs/04_resource_learning_plane_v2_2_nonredundant_concepts
```

The configuration used by the current 06 lineage is:

```powershell
python src/04_train_fused_conditional.py `
  --features outputs/04_resource_learning_plane_v2_2_nonredundant_concepts/04_prediction_safe_features.jsonl `
  --supervision outputs/04_resource_learning_plane_v2_2_nonredundant_concepts/04_resource_supervision.jsonl `
  --output-dir outputs/04_el_fafcor_v2_2_smoothing_005 `
  --experiment smoothing_005 `
  --realization-weighting equal `
  --partial-weight 1.0 `
  --class-weight sqrt_inverse `
  --ordinal-smoothing 0.05
```

### 05. Run audit-only error attribution

```powershell
python src/05_deepseek_error_harness.py `
  --transcripts transcripts/data `
  --stage01-cards outputs/01_evidence_cards_v1/evidence_cards.jsonl `
  --stage02 outputs/02_policy_trajectories_v1_deepseek_full/02_qualified_trajectories.jsonl `
  --stage03 outputs/03_vital_state_facts_v1/03_vital_state_facts.jsonl `
  --stage04-features outputs/04_resource_learning_plane_v2_2_nonredundant_concepts/04_prediction_safe_features.jsonl `
  --stage04-supervision outputs/04_resource_learning_plane_v2_2_nonredundant_concepts/04_resource_supervision.jsonl `
  --stage04-oof outputs/04_el_fafcor_v2_2_smoothing_005/04_oof_case_predictions.jsonl `
  --env-file src/.env `
  --output-dir outputs/05_deepseek_error_harness_v1
```

Stage 05b transmits deidentified dialogue and original gold ESI to the configured external API for audit. Run it only when data governance and API authorization permit this transfer.

```powershell
python src/05b_gold_esi_observability_harness.py `
  --transcripts transcripts/data `
  --stage01 outputs/01_evidence_cards_v1/evidence_cards.jsonl `
  --stage02 outputs/02_policy_trajectories_v1_deepseek_full/02_qualified_trajectories.jsonl `
  --stage03 outputs/03_vital_state_facts_v1/03_vital_state_facts.jsonl `
  --stage05-dir outputs/05_deepseek_error_harness_v1 `
  --output-dir outputs/05b_gold_esi_observability_full
```

### 06. Reconcile policy paths and review decisions

```powershell
python src/06_reconcile_and_loop_v2.py `
  --transcripts transcripts/data `
  --stage02 outputs/02_policy_trajectories_v1_deepseek_full/02_qualified_trajectories.jsonl `
  --stage03 outputs/03_vital_state_facts_v1/03_vital_state_facts.jsonl `
  --stage04-oof outputs/04_el_fafcor_v2_2_smoothing_005/04_oof_case_predictions.jsonl `
  --stage05-dir outputs/05_deepseek_error_harness_v1 `
  --stage05b-dir outputs/05b_gold_esi_observability_full `
  --output-dir outputs/06_policy_loop_closure_v2_gold_observability
```

## Current Verified Snapshot

The checked local artifacts report:

- 00: 1,010 realizations, 541 cases, zero structural or leakage-gate failures, and 40 quarantined measurements.
- 01: 16,798 evidence cards; zero forbidden leakage, question/process positive triggers, or invalid-measurement consumption.
- 02: 725 bounded LLM candidates, zero API fallbacks and evidence-ID hallucinations, and 154 validated Step B policy signals.
- 03: 5,047 vital observations, zero invalid-measurement consumption, and eight official SpO2 danger-zone signals.
- 04: grouped case-level OOF Macro-F1 `0.464`, balanced accuracy `0.480`, QWK `0.224`, and log loss `0.903` for the current 06 lineage.
- 06: engineering gates pass, but automatic coverage is `23.7%` and `clinical_release_ready` is `false`.

The 06 observable-target sensitivity result (ESI 2-5 Macro-F1 `0.587`) is development-only and excludes unassessable labels. It must not be presented as an unbiased clinical gold-standard result.

## Known Limitations

- Stage 04 does not pass its configured downstream-readiness gate; resource labels are ESI-derived weak supervision rather than adjudicated observed resource sets.
- Stage 06 deliberately sends most cases to review; no ESI 1 case is automatically released in the current snapshot.
- The clinical atom registry is closed and English-specific. Qwen and NLI are bounded ambiguity resolvers, not open clinical extractors.
- HR and respiratory-rate Decision Point D rules are not assessable when age is unavailable.
- Current counts and several release gates are tied to the checked 1,010-realization snapshot.
- External API stages require an explicit data-governance decision. Generated caches can contain dialogue excerpts and must not be committed.
- No prospective validation, clinician inter-rater study, or deployment evaluation has been completed.

## Upload Safety

The supplied `.gitignore` excludes:

- all `.env` files except examples;
- transcripts and generated outputs;
- external API caches and private audit maps through the ignored output tree;
- legacy `project/` and `research/` trees;
- large model and table artifacts.

Before publishing, review dataset licenses and decide whether any sample transcript can legally be redistributed. No license is currently assigned to this repository; add one only after confirming the intended reuse terms.

