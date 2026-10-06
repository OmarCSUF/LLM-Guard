# NIST AI RMF 1.0 mapping

Framework: NIST AI 100-1, Artificial Intelligence Risk Management Framework (AI RMF 1.0), January 2023.

Statements: AI RMF Core tables of NIST AI 100-1, quoted verbatim and checked against the PDF text on 23 September 2026.

This is a self-assessment of a research prototype against the subcategories of the AI RMF Core that the work bears on, not a conformance claim; the framework is voluntary and has no conformance procedure. Each row quotes the subcategory verbatim, names the repository files that bear on it and states the gap. Subcategories not listed are not claimed. The source of this page is `config/nist_ai_rmf_mapping.yaml`; `scripts/nist_mapping_report.py` checks every id against the 72 in the Core, every evidence path against the tree, and renders this file.

| Status | Meaning | Count |
|---|---|---|
| implemented | the statement holds for this system as built | 4 |
| partial | something in the repository bears on it, and the gap is named | 11 |
| not_implemented | nothing in the repository bears on it | 2 |
| not listed | not claimed | 55 of 72 |

## Audit records

With `logging.nist_mapping: true`, every audit record carries the ids below for its event, so a line in `logs/audit.jsonl` says which subcategories it is evidence for. Only ids with status implemented or partial may appear here.

| Event | Written by | Tagged with |
|---|---|---|
| `l1_decision` | Layer 1, `layer1/detector.py`, serving path only | GOVERN 4.3, MANAGE 4.1 |
| `l2_judgement` | Layer 2, `layer2/layer2_judge.py`, when `record=True` | GOVERN 4.3, MANAGE 4.1, MEASURE 2.9 |
| `l3_mitigation` | Layer 3, `layer3/layer3_mitigator.py`, when `record=True` | GOVERN 4.3, MANAGE 1.3, MANAGE 4.1 |

## GOVERN

| Subcategory | Statement | Status | Evidence | Gap |
|---|---|---|---|---|
| GOVERN 1.5 | Ongoing monitoring and periodic review of the risk management process and its outcomes are planned and organizational roles and responsibilities clearly defined, including determining the frequency of periodic review. | not_implemented | none | A research prototype with no organisation behind it: no roles, no review cadence and no risk management process to review. |
| GOVERN 4.3 | Organizational practices are in place to enable AI testing, identification of incidents, and information sharing. | partial | `scripts/run_evaluation.py`: repeatable benchmark evaluation with bootstrap intervals<br>`scripts/run_red_team.py`: repeatable adversarial testing, resumable, every variant stored<br>`scripts/run_ablation.py`: replayed ablation over stored signals<br>`utils/logger.py`: one record per decision at every layer, from which an incident can be identified after the fact | Testing is enabled and was done; no practice identifies incidents from the log while the system runs, and nothing is shared outside the thesis. The offline test suites and the numbered patch series live outside this repository. |
| GOVERN 6.1 | Policies and procedures are in place that address AI risks associated with third-party entities, including risks of infringement of a third-party's intellectual property or other rights. | partial | `artifacts/layer2/Qwen_Model_Justification.md`: choice of the third-party judge model and the alternatives considered | The judge model is released under a research licence, which the thesis records as a bar to release or commercial use; dataset licences are recorded at fetch time. Neither is a policy or a procedure. |

## MAP

| Subcategory | Statement | Status | Evidence | Gap |
|---|---|---|---|---|
| MAP 1.1 | Intended purposes, potentially beneficial uses, context-specific laws, norms and expectations, and prospective settings in which the AI system will be deployed are understood and documented. | partial | `README.md`: intended purpose, the three-layer design and the hardware target | Purpose and setting are documented; context-specific laws, norms and expectations are not addressed. |
| MAP 5.1 | Likelihood and magnitude of each identified impact (both potentially beneficial and harmful) based on expected use, past uses of AI systems in similar contexts, public incident reports, feedback from those external to the team that developed or deployed the AI system, or other data are identified and documented. | partial | `redteam/red_teamer.py`: bypass rate per attack type, with the mitigation served, as the measured likelihood of a harmful request passing<br>`docs/leak_codebook.md`: scoring rules that grade what a served rewrite supplies, as a measure of magnitude | Likelihood comes from one seed set and seven attack types; magnitude is graded by one reviewer so far. Past uses, public incident reports and external feedback are not used. |

## MEASURE

| Subcategory | Statement | Status | Evidence | Gap |
|---|---|---|---|---|
| MEASURE 1.1 | Approaches and metrics for measurement of AI risks enumerated during the map function are selected for implementation starting with the most significant AI risks. | implemented | `evaluation/metrics.py`: harmful-allowed and safe-blocked rates, AUROC, calibration, paired bootstrap intervals<br>`docs/validation_policy.md`: which set selected each metric and threshold, and which set reports it | none |
| MEASURE 2.5 | The AI system to be deployed is demonstrated to be valid and reliable. Limitations of the generalizability beyond the conditions under which the technology was developed are documented. | implemented | `evaluation/pipeline_evaluator.py`: evaluation on four external benchmarks<br>`evaluation/metrics.py`: bootstrap intervals on every reported rate<br>`docs/validation_policy.md`: selection and reporting sets kept apart; decisions that were made on the reporting set are listed as such | Reliability across training seeds was measured for Layer 1 only, where the same recipe moves AUROC by 0.04 between seeds. The limits of generalisation are in the thesis, not in this repository. |
| MEASURE 2.6 | The AI system is evaluated regularly for safety risks – as identified in the map function. The AI system to be deployed is demonstrated to be safe, its residual negative risk does not exceed the risk tolerance, and it can fail safely, particularly if made to operate beyond its knowledge limits. | partial | `layer3/mitigator.py`: refuse is the default mode; an uncertain verdict, a refuse-always category or a high severity all refuse<br>`layer3/layer3_mitigator.py`: safe_rewrite falls back to refuse when no Guard-LLM is loaded, when every candidate fails or when the rewrite check withholds it<br>`redteam/red_teamer.py`: an exception during classification counts as a bypass, so a crash cannot improve a rate | Evaluated once, not regularly. No risk tolerance was stated in advance, so residual risk is reported rather than compared with a bound. |
| MEASURE 2.7 | AI system security and resilience – as identified in the map function – are evaluated and documented. | implemented | `redteam/red_teamer.py`: adversarial evaluation, rates over distinct variants, every variant stored with the mitigation served<br>`redteam/attacks.py`: seven attack types, each labelled by whether it provably keeps the harmful request<br>`layer1/normalise.py`: the resilience measure adopted in response, input normalisation and payload views | One seed set and one harm taxonomy; the two model-written attack types share the judge model, so the attacker is no stronger than the defender. |
| MEASURE 2.9 | The AI model is explained, validated, and documented, and AI system output is interpreted within its context – as identified in the map function – to inform responsible use and governance. | partial | `layer2/layer2_judge.py`: the judge's decision, category, severity and rationale are written to the audit record<br>`layer1/detector.py`: each decision exposes its three signals, the fused and calibrated scores and the normalisation applied<br>`scripts/run_ablation.py`: which signals carry information, measured by removing them | The rationale is model-written text, not a validated explanation. The Layer-1 classifier is explained only by its inputs. |
| MEASURE 2.11 | Fairness and bias – as identified in the map function – are evaluated and results are documented. | partial | `evaluation/l2_variants.py`: safe prompts rewritten with grammatical features typical of non-native English, compared prompt by prompt with the originals | A grammatical proxy for non-native writing, not a sample of it; no other attribute was examined. |
| MEASURE 3.3 | Feedback processes for end users and impacted communities to report problems and appeal system outcomes are established and integrated into AI system evaluation metrics. | not_implemented | none | No feedback or appeal channel exists. The escalated_to_human flag is set by Layer 3 and consumed by nothing. |
| MEASURE 4.2 | Measurement results regarding AI system trustworthiness in deployment context(s) and across the AI lifecycle are informed by input from domain experts and relevant AI actors to validate whether the system is performing consistently as intended. Results are documented. | partial | `docs/leak_codebook.md`: manual review protocol for served rewrites, with a second reviewer and pre-committed agreement statistics<br>`scripts/leak_reliability.py`: agreement, clustering and sensitivity computations for that review | One reviewer so far, and no domain expert. The second review is planned, not done. |

## MANAGE

| Subcategory | Statement | Status | Evidence | Gap |
|---|---|---|---|---|
| MANAGE 1.3 | Responses to the AI risks deemed high priority, as identified by the map function, are developed, planned, and documented. Risk response options can include mitigating, transferring, avoiding, or accepting. | implemented | `layer3/mitigator.py`: mode selection rules, refuse-always categories, severity threshold, escalation refused<br>`config/config.yaml`: layer3.mitigation.default_mode is refuse<br>`config/safety_templates.yaml`: refusal templates per category and the constraint patterns<br>`layer1/normalise.py`: the response to obfuscation attacks found by the red team | The options used are mitigate and avoid. Residual bypasses are documented; no explicit accepted-risk statement exists. |
| MANAGE 2.4 | Mechanisms are in place and applied, and responsibilities are assigned and understood, to supersede, disengage, or deactivate AI systems that demonstrate performance or outcomes inconsistent with intended use. | partial | `layer3/layer3_mitigator.py`: when every rewrite candidate fails, the output is a fixed notice and escalated_to_human is set; without a Guard-LLM the mode falls back to refuse<br>`layer3/mitigator.py`: an uncertain judge verdict is refused rather than generated for | The mechanisms disengage generation for one prompt. There is no operator switch, no responsibility assigned, and the escalation flag is received by nothing. |
| MANAGE 4.1 | Post-deployment AI system monitoring plans are implemented, including mechanisms for capturing and evaluating input from users and other relevant AI actors, appeal and override, decommissioning, incident response, recovery, and change management. | partial | `utils/logger.py`: every decision at every layer is captured as a JSON line, with a hash that joins one prompt's three records<br>`layer2/distillation.py`: confident judge labels are captured to a pool | A log and a pool are capture mechanisms, not a plan. There is no review cadence, appeal, override, decommissioning, incident response or change management, and nothing reads the pool. |
| MANAGE 4.3 | Incidents and errors are communicated to relevant AI actors, including affected communities. Processes for tracking, responding to, and recovering from incidents and errors are followed and documented. | partial | `redteam/red_teamer.py`: every bypass and every leaked rewrite is written to a failure log with the output served<br>`scripts/run_red_team.py`: per-variant rows stream to disk and survive an interruption | Failures are recorded, not communicated to anyone, and there is no incident process. |
