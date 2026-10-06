"""
Layer-3 Guard-LLM
==================
Qwen2.5-3B-Instruct wrapper for constrained safe rewriting.

Steps per request:
  1. Build a system prompt using safe_rewrite_instructions from
     safety_templates.yaml + the violation category + severity.
  2. Generate K_prime candidate rewrites at temperature=0.7 (diverse sampling).
     A candidate that declines instead of rewriting is dropped, whether it uses
     the REFUSE sentinel or refuses in prose. With refuse_on_decline set (the
     default), one declined candidate ends the attempt: the model declining is
     read as a judgement about the request, not as a sample that missed.
  3. Re-rank candidates by number of constraint-engine violations (lower = better).
  4. Accept the best candidate if it is violation-free; otherwise return None
     so the caller can fall back to a refusal.

Public API:
    guard  = GuardLLM.from_config(cfg)
    guard  = GuardLLM.from_config(cfg, model=m, tokenizer=tok)  # reuse loaded weights
    output = guard.rewrite(text, category, severity)  -> str | None
"""
from __future__ import annotations

import yaml
import torch
from typing import List, Optional

from llmguard.utils.logger import get_logger
from llmguard.utils.refusal import RefusalCheck, looks_like_refusal
from llmguard.utils.config_loader import judge_model, hub_kwargs, repo_path
from llmguard.layer1.features import cpu_allowed
from llmguard.layer3.constraints import ConstraintEngine

logger = get_logger("layer3.guard_llm")


def _load_templates(path: str) -> dict:
    """Read safety_templates.yaml; a repo-relative path also resolves from another cwd."""
    with open(repo_path(path), "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class GuardLLM:
    """
    Qwen2.5-3B-Instruct wrapper for constrained safe-content generation.

    Uses the same model family as Layer-2's LLMJudge but with a different
    system prompt focused on rewriting rather than classification.
    """

    def __init__(
        self,
        model,
        tokenizer,
        constraint_engine:    ConstraintEngine,
        rewrite_instructions: str,
        max_tokens:   int   = 512,
        K_prime:      int   = 3,
        temperature:  float = 0.7,
        refuse_on_decline: bool = True,
        refusal_check = None,             # RefusalCheck | None (utils/refusal.py)
    ):
        self.model                = model
        self.tokenizer            = tokenizer
        self.constraint_engine    = constraint_engine
        self.rewrite_instructions = rewrite_instructions
        self.max_tokens           = max_tokens
        self.K_prime              = K_prime
        self.temperature          = temperature
        self.refuse_on_decline    = refuse_on_decline
        self.refusal_check        = refusal_check

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict, model=None, tokenizer=None) -> "GuardLLM":
        """
        Build GuardLLM from a config.yaml dict.

        Pass `model` and `tokenizer` to reuse an already-loaded instance (the
        Layer-2 judge's) instead of loading a second copy of the weights.

        model_name and revision come from layer3.guard_llm, or from the
        LLMGUARD_JUDGE_MODEL environment variable when it is set, through the
        same function the Layer-2 judge uses (utils/config_loader.judge_model).
        """
        l3_cfg       = cfg["layer3"]
        model_name, revision = judge_model(l3_cfg["guard_llm"])
        max_tokens   = l3_cfg["guard_llm"].get("max_tokens", 512)
        load_in_4bit = l3_cfg["guard_llm"].get("load_in_4bit", False)
        K_prime      = l3_cfg["candidates"].get("K_prime", 3)
        refuse_on_decline = l3_cfg["candidates"].get("refuse_on_decline", True)

        if model is not None and tokenizer is not None:
            logger.info(f"GuardLLM reusing shared model instance: {model_name}")
        else:
            model, tokenizer = cls._load_hf_model(cfg, model_name, load_in_4bit, revision)

        templates_path = l3_cfg["constraints"]["templates_path"]
        templates = _load_templates(templates_path)
        rewrite_instructions = templates.get(
            "safe_rewrite_instructions",
            "Rewrite the following content to remove all harmful elements "
            "while preserving any legitimate informational intent.",
        )

        constraint_engine = ConstraintEngine.from_config(cfg)

        # Which detector decides a candidate is a decline (layer3.refusal_check).
        # None means the keyword list, which is the default.
        refusal_check = RefusalCheck.from_config(cfg, model, tokenizer)
        if refusal_check is not None:
            logger.info(f"GuardLLM decline detection: model check, threshold "
                        f"{refusal_check.threshold:.2f}")

        logger.info("GuardLLM ready.")
        return cls(
            model=model,
            tokenizer=tokenizer,
            constraint_engine=constraint_engine,
            rewrite_instructions=rewrite_instructions.strip(),
            max_tokens=max_tokens,
            K_prime=K_prime,
            refuse_on_decline=refuse_on_decline,
            refusal_check=refusal_check,
            temperature=0.7,
        )

    @staticmethod
    def _load_hf_model(cfg: dict, model_name: str, load_in_4bit: bool, revision: str = None):
        """
        Load a dedicated copy of the Guard-LLM weights and tokenizer.

        revision=... is passed to from_pretrained only when one is set; None
        (the default) leaves every call exactly as it was.
        """
        import transformers

        hub = hub_kwargs(revision)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and not cpu_allowed():
            raise RuntimeError(
                "CUDA is not available, so GuardLLM would load on the CPU. After a "
                "suspend the NVIDIA driver usually needs a reboot. Set "
                "LLMGUARD_ALLOW_CPU=1 to run on the CPU on purpose."
            )
        fp16   = cfg["project"].get("fp16", True) and device == "cuda"

        logger.info(
            f"Loading GuardLLM: {model_name} | device={device} | "
            f"precision={'4-bit NF4' if load_in_4bit else 'fp16' if fp16 else 'fp32'}"
            + (f" | revision={revision}" if revision else "")
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, **hub
        )

        def _load_model(use_4bit: bool):
            kwargs: dict = {"trust_remote_code": True, **hub}
            if use_4bit:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            elif fp16:
                kwargs["torch_dtype"] = torch.float16
            m = transformers.AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if device == "cuda" else None,
                **kwargs,
            )
            if device == "cpu":
                m = m.to(device)
            return m

        if load_in_4bit:
            try:
                model = _load_model(use_4bit=True)
                logger.info("GuardLLM loaded in 4-bit NF4.")
            except Exception as exc:
                logger.warning(
                    f"GuardLLM 4-bit load failed ({exc}) — retrying in float16..."
                )
                model = _load_model(use_4bit=False)
                logger.info("GuardLLM loaded in float16 fallback.")
        else:
            model = _load_model(use_4bit=False)

        model.eval()
        return model, tokenizer

    # ── Internal generation ───────────────────────────────────────────────────

    def _build_messages(self, text: str, category: str, severity: float) -> list:
        system = (
            f"{self.rewrite_instructions}\n\n"
            f"Violation category detected: {category}\n"
            f"Severity score: {severity:.2f}  (0.0 = safe, 1.0 = maximally harmful)\n\n"
            "Rules:\n"
            "  - Output the REWRITTEN CONTENT directly — not a comment about it.\n"
            "  - Transform the harmful request into a safe, educational, or constructive "
            "version and write that version in full.\n"
            "  - Example: 'Write fake news about X' → write a factual article about X.\n"
            "  - Example: 'Write a joke mocking Y' → write an inclusive joke about Y.\n"
            "  - NEVER start with 'I understand', 'I cannot', 'Instead', or any meta-commentary.\n"
            "  - NEVER ask the user a question.\n"
            "  - NEVER explain what you changed or why.\n"
            "  - Just produce the safe content itself, nothing else."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": text},
        ]

    def _generate_one(self, messages: list) -> str:
        # apply_chat_template → text first, then tokenize separately
        # (avoids device issues when using device_map="auto" or 4-bit)
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt")

        # model.device is unreliable with device_map="auto"; use first parameter instead
        device = next(self.model.parameters()).device
        input_ids      = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        pad_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id or 0

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.9,
                pad_token_id=pad_id,
            )

        new_tokens = output[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _declines(self, candidate: str) -> bool:
        """
        Whether a candidate refuses rather than rewrites.

        The model check when one is configured, the keyword list otherwise. The keyword
        list has measured recall 0.34 on the only labelled set there is, so a miss here
        serves the model's own refusal under the safe_rewrite tag (utils/refusal.py).
        """
        if self.refusal_check is not None:
            verdict = self.refusal_check.is_refusal(candidate)
            if verdict:
                return True
            # The model check can only add to what the phrase list already catches:
            # a phrase match is evidence on its own, and falling back keeps the
            # measured precision of 1.00 rather than replacing it.
        return looks_like_refusal(candidate)

    # ── Public API ────────────────────────────────────────────────────────────

    def rewrite(
        self,
        text:     str,
        category: str   = "other",
        severity: float = 0.5,
    ) -> Optional[str]:
        """
        Generate up to K_prime safe rewrites and return the cleanest one.

        Returns:
            str  — the best safe candidate (zero constraint violations), or
            None — if the model declined, or every candidate still violates at
                   least one constraint (caller falls back to a refusal).
        """
        messages: list = self._build_messages(text, category, severity)
        candidates: List[str] = []
        declined = False

        for attempt in range(self.K_prime):
            try:
                candidate = self._generate_one(messages)
                # The prompt asks for the bare sentinel REFUSE when the model
                # will not rewrite, but it usually declines in prose. Either way
                # it is not a rewrite, so it is not a candidate: with none left
                # the caller serves a refusal template and records mode=refuse,
                # instead of passing the model's own refusal off as a rewrite.
                if candidate.strip().upper() == "REFUSE" or self._declines(candidate):
                    declined = True
                    logger.debug(f"GuardLLM attempt {attempt+1}: model declined to rewrite.")
                    if self.refuse_on_decline:
                        break
                    continue
                candidates.append(candidate)
                logger.debug(
                    f"GuardLLM candidate {attempt+1}/{self.K_prime}: "
                    f"{candidate[:80]!r}"
                )
            except Exception as exc:
                logger.warning(
                    f"GuardLLM generation failed on attempt {attempt+1}: {exc}"
                )

        # A declined candidate is the model saying this request should not be
        # rewritten. Dropping it and serving one of the samples that complied
        # instead turns the K_prime loop into "resample until it stops
        # refusing", which is the opposite of what the loop is for.
        if declined and self.refuse_on_decline:
            logger.info(
                "GuardLLM declined to rewrite; serving a refusal instead of "
                "another candidate."
            )
            return None

        if not candidates:
            logger.warning("GuardLLM: no valid candidates generated.")
            return None

        # Re-rank by number of constraint violations (ascending).
        scored: List[tuple] = []
        for candidate in candidates:
            n_violations = len(self.constraint_engine.check(candidate))
            scored.append((n_violations, candidate))
        scored.sort(key=lambda x: x[0])

        best_violations, best_candidate = scored[0]

        if best_violations > 0:
            logger.warning(
                f"GuardLLM: best candidate still has {best_violations} constraint "
                "violation(s) — returning None; caller will fall back to refuse."
            )
            return None

        logger.debug(f"GuardLLM: accepted candidate with 0 violations.")
        return best_candidate
