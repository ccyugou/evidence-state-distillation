# EvidenceState

Teacher-student evidence parsing and auditable state revision for multi-turn clinical dialogue.

## Research problem

Clinical dialogue contains short answers, omitted subjects, historical descriptions, facet updates, and corrections. A sentence such as "没有了" cannot be interpreted safely without the doctor's question and the previous symptom state.

This project separates language understanding from state execution. The language model determines what a question-answer pair means; the state tracker applies only the resulting structured operation. This avoids relying on an expanding collection of phrase rules while keeping every state change traceable to its source evidence.

## Pipeline

```text
IMCS-21 dialogue
    |
    v
00 Data contract and case-level audit
    |
    v
01 Evidence compiler
    |-- medical entity extraction
    |-- normalized symptom and source span
    |-- subject, time, assertion, and facet proposal
    |
    v
02 Ordered state revision
    |-- present / absent / resolved / recurrent
    |-- facet update without incorrectly deleting symptom existence
    |-- unresolved evidence remains HOLD
    |
    v
03 Clinical belief matrix
    |-- turn-level state snapshots
    |-- revision trajectory and evidence provenance
    `-- quality and unresolved-evidence indicators
```

### Stage 00: data contract

The raw IMCS-21 files are audited before modeling. Prediction text is separated from diagnosis, report, dialogue-act, BIO, and symptom-status annotations. Cases, rather than individual turns, are the unit of data isolation.

### Stage 01: evidence compilation

A RoBERTa token classifier extracts medical mentions. Each evidence card stores the normalized concept, original span, speaker, turn, and local context. Ambiguous question-answer semantics are represented with a closed schema instead of free-form generation.

### Stage 02: state revision

Evidence is applied in chronological order. Existence and facets are separate: "没有那么痛" may update severity while preserving pain as present, whereas "已经不痛了" may retract current existence. Subject, historical, unsupported, and unresolved evidence cannot silently overwrite the patient's current state.

### Stage 03: belief matrix

The tracker produces turn-level snapshots and a final clinical belief matrix. Every state contains its supporting evidence and ordered transition history, so the representation can be inspected or corrected without rereading the entire dialogue.

## Teacher-student semantic model

The learned semantic component replaces open-ended rule expansion for difficult question-answer cases.

1. Qwen3.5-9B proposes a structured state-edit interpretation under balanced candidate ordering.
2. DeepSeek independently judges the original dialogue packet.
3. Disagreements are adjudicated again from the raw packet; unresolved cases become `HOLD`.
4. Accepted multi-view labels form proxy/silver supervision.
5. Qwen3-4B is trained with QLoRA to reproduce the structured operation locally.

The student predicts:

- target binding;
- patient or non-patient subject;
- current or historical time;
- symptom existence effect;
- facet update effect;
- updated facet types;
- exact supporting quotes.

Large-model agreement is used as proxy semantic supervision, not as clinical expert gold.

## Experiment design

- Train and Dev are separated by clinical case.
- A locked split is not used for prompt, threshold, or model selection.
- The same Qwen3-4B base model is evaluated before and after QLoRA adaptation.
- Evaluation uses paired examples and case-level bootstrap confidence intervals.
- `HOLD` is an explicit unresolved result and never means that a symptom is absent.
- Dataset labels are not treated as gold state-edit operations.

## Development result

| Model | Full structured-state exact match |
| --- | ---: |
| Qwen3-4B zero-shot | 0.2780 |
| Qwen3-4B QLoRA student | 0.6051 |
| Paired improvement | +0.3270 |

The case-level bootstrap 95% confidence interval for the paired improvement was `[0.2871, 0.3672]`.

This result supports stable teacher-to-student transfer of common structured semantic operations. It does not establish clinician-level accuracy, diagnosis quality, or end-to-end clinical correctness.

## Code layout

```text
scripts/
|-- common/       shared schemas and utilities
|-- data/         Stage 00 data audit
|-- evidence/     Stage 01 evidence compilation
|-- states/       Stage 02 and Stage 03 state construction
|-- teacher/      Qwen3.5 and DeepSeek proxy teacher
`-- student/      Qwen3-4B distillation and evaluation
```

IMCS-21 data, generated labels, API credentials, model weights, adapters, and experiment outputs are not included. Local paths and secrets are configured through `.env`; see `.env.example`.

## Scope

EvidenceState studies structured evidence interpretation and longitudinal state revision. It does not provide diagnosis, treatment recommendation, or ESI classification. Dataset and model use must follow their original licenses and privacy requirements.
