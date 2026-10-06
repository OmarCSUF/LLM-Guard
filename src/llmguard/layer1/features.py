"""
Layer-1 Feature Signals
========================
Three complementary signals used by the logistic stacker:

  (a) CurvatureFeature  — perplexity + curvature under decoding perturbations
                          (DetectGPT-style, GPU-accelerated via logit scaling)
  (b) LogRankFeature    — log-percentile rank of each token
                          (DetectLLM-style, memory-efficient GPU ranking)
  (c) ClassifierFeature — score from distilled RoBERTa+LoRA classifier
                          (trained from Judge labels)

All features return a scalar in [0, 1] where
  → 0  means "likely safe / allow"
  → 1  means "likely harmful / mitigate"

GPU changes vs. CPU version:
  - Auto-detects CUDA; uses float16 on GPU, float32 on CPU
  - Mixed-precision autocast in all forward passes
  - LogRankFeature uses comparison-based ranking (avoids O(T×V) argsort)
  - Shared model between curvature and log-rank (single load)
  - torch.cuda.empty_cache() on explicit cleanup

References:
  [Mitchell et al., 2023] DetectGPT
  [Bao et al., 2024]      Fast-DetectGPT
  [Su et al., 2023]       DetectLLM
  [Gehrmann et al., 2019] GLTR
"""

from __future__ import annotations

import math
import os
import torch
import torch.nn.functional as F
from contextlib import contextmanager
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional
from llmguard.utils.config_loader import hub_kwargs
from llmguard.utils.logger import get_logger

logger = get_logger("layer1.features")

# Hub ids the recorded runs resolved from the local HF cache. The config
# (layer1.scoring_lm, layer1.classifier) names them and pins their commits;
# these defaults keep hand-built features and older scripts on the same models.
DEFAULT_SCORING_LM = "distilgpt2"
DEFAULT_BACKBONE = "roberta-base"


# ─────────────────────────────────────────────────────────────────────────────
# Device & dtype helpers
# ─────────────────────────────────────────────────────────────────────────────

def cpu_allowed() -> bool:
    """True when LLMGUARD_ALLOW_CPU=1 (or true) permits running models on the CPU."""
    return os.environ.get("LLMGUARD_ALLOW_CPU", "").strip().lower() in ("1", "true")


def resolve_device(device: str = "auto") -> str:
    """
    Resolve "auto" to "cuda" or "cpu" based on availability.
    Logs the selected device so the user always knows what is running.

    "auto" refuses to fall back to the CPU unless LLMGUARD_ALLOW_CPU=1: a GPU
    that disappears (for example after a laptop suspend) would otherwise turn
    a 20-minute run into hours without any error. Pass "cpu" to use the CPU.
    """
    if device == "auto":
        if torch.cuda.is_available():
            selected = "cuda"
        elif cpu_allowed():
            selected = "cpu"
        else:
            raise RuntimeError(
                "CUDA is not available, so device 'auto' would silently run on the CPU. "
                "After a suspend the NVIDIA driver usually needs a reboot. To run on the "
                "CPU on purpose, set project.device: \"cpu\" or LLMGUARD_ALLOW_CPU=1."
            )
    else:
        selected = device
    logger.info(f"Device resolved: {selected}"
                + (f" ({torch.cuda.get_device_name(0)})" if selected == "cuda" else ""))
    return selected


def model_dtype(device: str, fp16: bool = True) -> torch.dtype:
    """Return float16 on GPU (if fp16=True), float32 otherwise."""
    if device == "cuda" and fp16:
        return torch.float16
    return torch.float32


@contextmanager
def autocast_ctx(device: str):
    """Context manager: enables autocast on GPU, no-op on CPU."""
    if device == "cuda":
        with torch.amp.autocast("cuda"):
            yield
    else:
        yield


# ─────────────────────────────────────────────────────────────────────────────
# (a) Curvature / Perplexity Feature
# ─────────────────────────────────────────────────────────────────────────────

class CurvatureFeature:
    """
    Computes a curvature-based anomaly score.

    Algorithm:
      1. Score text x under base LM at default temperature → NLL(x)
      2. Score x under M perturbed temperature scales {T_m} → NLL_m(x)
      3. fcurv(x) = mean_m[ NLL_m(x) - NLL(x) ]

    Intuition:
      Model-generated text tends to sit near a local log-prob maximum.
      Perturbing the scoring temperature changes NLL less for model-text
      than for human-text, giving a discriminative curvature signal.

    GPU notes:
      - Uses float16 on CUDA to halve VRAM usage and speed up inference.
      - All forward passes run under autocast for numerical stability.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SCORING_LM,
        num_perturbations: int = 10,
        temperature_range: tuple = (0.5, 1.5),
        max_length: int = 512,
        device: str = "auto",
        fp16: bool = True,
        revision: Optional[str] = None,
    ):
        """
        revision : hub commit to pin for model_name (layer1.scoring_lm.revision).
                   None, the default, calls from_pretrained as before.
        """
        self.device = resolve_device(device)
        self.fp16 = fp16
        self.max_length = max_length
        self.num_perturbations = num_perturbations
        self.model_name = model_name
        self.revision = revision or None

        # Evenly spaced temperatures across the range
        if num_perturbations > 1:
            step = (temperature_range[1] - temperature_range[0]) / (num_perturbations - 1)
            self.temperatures = [
                temperature_range[0] + i * step for i in range(num_perturbations)
            ]
        else:
            self.temperatures = [1.0]

        dtype = model_dtype(self.device, fp16)
        logger.info(f"Loading scoring LM: {model_name} (dtype={dtype})"
                    + (f" revision={revision}" if revision else ""))

        hub = hub_kwargs(self.revision)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **hub)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            **hub,
        ).to(self.device)
        self.model.eval()

        # distilgpt2 / gpt2 family has no pad token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(
            f"CurvatureFeature ready. "
            f"Perturbations={num_perturbations}, "
            f"T range={temperature_range}, device={self.device}"
        )

    @torch.no_grad()
    def _compute_nll(self, input_ids: torch.Tensor, temperature: float = 1.0) -> float:
        """
        Average NLL per token under a given temperature scaling.

        Args:
            input_ids: [1, T] token ids on self.device
            temperature: logit temperature divisor (>1 flattens, <1 sharpens)

        Returns:
            scalar float (NLL per token)
        """
        if input_ids.shape[1] < 2:
            return 0.0

        with autocast_ctx(self.device):
            outputs = self.model(input_ids)
            logits = outputs.logits[:, :-1, :].float()   # cast to float32 for stability
            targets = input_ids[:, 1:]

            scaled_logits = logits / temperature
            log_probs = F.log_softmax(scaled_logits, dim=-1)

            token_log_probs = log_probs.gather(
                dim=-1, index=targets.unsqueeze(-1)
            ).squeeze(-1)   # [1, T-1]

        return -token_log_probs.mean().item()

    @torch.no_grad()
    def compute(self, text: str) -> float:
        """
        Returns fcurv(x) normalised to [0, 1] via sigmoid.
        Higher → more anomalous → more likely to mitigate.
        """
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        input_ids = enc["input_ids"]

        if input_ids.shape[1] < 2:
            return 0.5  # too short to score reliably

        base_nll = self._compute_nll(input_ids, temperature=1.0)
        perturbed_nlls = [
            self._compute_nll(input_ids, temperature=t)
            for t in self.temperatures
        ]
        fcurv = sum(n - base_nll for n in perturbed_nlls) / len(perturbed_nlls)

        # sigmoid normalisation: large positive curvature → higher score
        score = torch.sigmoid(torch.tensor(float(fcurv), dtype=torch.float32)).item()
        return float(score)

    def compute_batch(self, texts: List[str]) -> List[float]:
        return [self.compute(t) for t in texts]


# ─────────────────────────────────────────────────────────────────────────────
# (b) Log-Rank Feature
# ─────────────────────────────────────────────────────────────────────────────

class LogRankFeature:
    """
    Computes the average log-percentile-rank of each token.

    Algorithm (DetectLLM / GLTR-inspired):
      For each token w_t:
        r_t  = number of vocabulary tokens with logit >= logit(w_t)
               (1-indexed rank; 1 = most likely)
        p_t  = r_t / |V|        (percentile; higher → less likely token)
      frank(x) = (1/T) * Σ log(1 + p_t)

    GPU memory fix (vs. original version):
      The old implementation used double argsort: O(T × V × log V) time and
      O(T × V) extra memory — problematic for long sequences on GPU.

      This version uses a comparison-based rank:
        rank(w_t) = (logits[t] >= logit_of_w_t).sum()
      which requires only O(1) extra memory per token position and runs
      efficiently as a broadcasted GPU comparison.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SCORING_LM,
        max_length: int = 512,
        device: str = "auto",
        fp16: bool = True,
        shared_model=None,
        shared_tokenizer=None,
        revision: Optional[str] = None,
    ):
        """
        Pass shared_model / shared_tokenizer from a CurvatureFeature instance
        to avoid loading the same LM twice. revision pins the hub commit when
        the LM is loaded here; None (the default) is the old call.
        """
        self.max_length = max_length

        if shared_model is not None and shared_tokenizer is not None:
            self.model = shared_model
            self.tokenizer = shared_tokenizer
            # Infer device and fp16 from shared model
            self.device = next(shared_model.parameters()).device.type
            self.fp16 = next(shared_model.parameters()).dtype == torch.float16
            logger.info("LogRankFeature: reusing shared LM.")
        else:
            self.device = resolve_device(device)
            self.fp16 = fp16
            dtype = model_dtype(self.device, fp16)
            logger.info(f"Loading scoring LM: {model_name} (dtype={dtype})"
                        + (f" revision={revision}" if revision else ""))
            hub = hub_kwargs(revision)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, **hub)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=dtype, **hub
            ).to(self.device)
            self.model.eval()
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def compute(self, text: str) -> float:
        """
        Returns frank(x) normalised to [0, 1].
        Higher → tokens are on average less likely → suspicious.
        """
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        input_ids = enc["input_ids"]

        if input_ids.shape[1] < 2:
            return 0.5

        with autocast_ctx(self.device):
            outputs = self.model(input_ids)
            # Cast to float32 for stable rank computation
            logits = outputs.logits[:, :-1, :].float()   # [1, T-1, V]
            targets = input_ids[:, 1:]                   # [1, T-1]

        # Logit value of each actual target token: [1, T-1, 1]
        target_logits = logits.gather(
            dim=-1, index=targets.unsqueeze(-1)
        )   # [1, T-1, 1]

        # Rank = number of vocabulary entries with logit >= target logit
        # Comparison broadcast: [1, T-1, V] >= [1, T-1, 1] → bool → sum over V
        token_ranks = (logits >= target_logits).sum(dim=-1).float()  # [1, T-1]

        vocab_size = logits.shape[-1]
        percentiles = token_ranks / vocab_size           # p_t ∈ (0, 1]

        frank = torch.mean(torch.log(1.0 + percentiles)).item()

        # frank ∈ [log(1), log(2)] ≈ [0, 0.693]. Normalise to [0, 1].
        normalised = min(frank / math.log(2.0), 1.0)
        return float(normalised)

    def compute_batch(self, texts: List[str]) -> List[float]:
        return [self.compute(t) for t in texts]


# ─────────────────────────────────────────────────────────────────────────────
# (c) Distilled Classifier Feature
# ─────────────────────────────────────────────────────────────────────────────

class ClassifierFeature:
    """
    Wraps the fine-tuned RoBERTa+LoRA binary classifier.

    Outputs g(x) ∈ [0, 1] = P(allow), the probability of label 1: the classifier
    is trained on the pool labels (1 = allow, 0 = mitigate).
      → 1 = allow (safe)
      → 0 = mitigate (harmful)
    The logistic stacker is fitted on this value, so it is not flipped here;
    use 1 - g(x) wherever a harm score is needed on its own.

    Trained on the Layer-1 pool labels via scripts/train_layer1.py.
    Falls back to 0.5 (uninformative) if no checkpoint is available.

    GPU notes:
      - Loads model in float16 on CUDA.
      - Inference uses autocast for speed.
      - Supports batched inference for efficiency.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        backbone: str = DEFAULT_BACKBONE,
        max_length: int = 512,
        device: str = "auto",
        fp16: bool = True,
        backbone_revision: Optional[str] = None,
    ):
        """
        backbone_revision : hub commit to pin for the backbone
                            (layer1.classifier.backbone_revision). None, the
                            default, calls from_pretrained as before.
        """
        self.device = resolve_device(device)
        self.fp16 = fp16
        self.max_length = max_length
        self.backbone = backbone
        self.backbone_revision = backbone_revision or None
        self.model = None
        self.tokenizer = None
        self._loaded = False

        if checkpoint_path is not None:
            self._load(checkpoint_path)
        else:
            logger.warning(
                "ClassifierFeature: no checkpoint provided. "
                "Returning 0.5 for all inputs. "
                "Run scripts/train_layer1.py first."
            )

    def _load(self, checkpoint_path: str) -> None:
        from transformers import AutoModelForSequenceClassification
        from peft import PeftModel

        logger.info(f"Loading classifier from: {checkpoint_path}")
        dtype = model_dtype(self.device, self.fp16)

        hub = hub_kwargs(self.backbone_revision)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, **hub)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.backbone,
            num_labels=2,
            torch_dtype=dtype,
            **hub,
        )
        self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info(f"Classifier loaded on {self.device} (dtype={dtype}).")

    @torch.no_grad()
    def compute(self, text: str) -> float:
        """Returns g(x) = P(allow) ∈ [0, 1]."""
        if not self._loaded:
            return 0.5

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)

        with autocast_ctx(self.device):
            outputs = self.model(**enc)

        probs = F.softmax(outputs.logits.float(), dim=-1)
        return float(probs[0, 1].item())   # P(allow): label 1 = allow

    @torch.no_grad()
    def compute_batch(self, texts: List[str], batch_size: int = 64) -> List[float]:
        """Batched inference — significantly faster than one-by-one on GPU."""
        if not self._loaded:
            return [0.5] * len(texts)

        all_scores: List[float] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i: i + batch_size]
            enc = self.tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            ).to(self.device)

            with autocast_ctx(self.device):
                outputs = self.model(**enc)

            probs = F.softmax(outputs.logits.float(), dim=-1)
            all_scores.extend(probs[:, 1].tolist())   # P(allow) for each

        return all_scores
