"""
Grammatical variation for the RQ4 over-refusal parity probe.

Rewrites a *benign* prompt with grammatical features typical of writers whose
first language is not English, leaving every content word in place. The
question it serves is whether the guard blocks ordinary, safe requests more
often when they carry these features. Callers must pass only prompts labelled
safe; scripts/make_l2_benchmarks.py enforces that.

Two profiles:

  general   features common across many first languages: dropped articles,
            preposition substitution, subject-verb agreement, plural -s after
            a quantifier, and "discuss about"-style preposition insertion
  arabic    features associated with Arabic as a first language: the definite
            article in place of a/an, present-tense copula omission, and a
            resumptive pronoun after relative "that", plus the preposition,
            agreement and plural features

The inventory follows descriptions in the learner-English literature, such as
the chapter on Arabic speakers in Swan and Smith, Learner English (2nd ed.,
Cambridge University Press, 2001). It is a proxy and should be reported as
one: real non-native writing also differs in vocabulary and phrasing, which
these rules do not model.

Edits are recorded per prompt so the manifest can report coverage, and output
is deterministic for a given prompt id and profile.
"""
import hashlib
import random
import re
from typing import Dict, List, Tuple

PROFILES = ("general", "arabic")

# Lines that look like code, markup or a link are left alone: ToxicChat in
# particular contains pasted code, where an article edit would be nonsense.
_CODE = re.compile(r"""
    ```                 # a fenced block
  | ^[ \t]{2,}\S         # an indented line, which is how pasted code arrives
  | \w+\(               # a call: print(...), range(...)
  | \[[^\]]*\]          # a subscript or index
  | [{}<>=]             # braces, comparisons, assignment
  | \b(def|import|class|return|SELECT|INSERT)\b
  | https?://|www\.
""", re.VERBOSE)
_PROTECTED_TOKEN = re.compile(r"[/@_\\=]|https?:")

_QUANT = (r"two|three|four|five|six|seven|eight|nine|ten|twenty|"
          r"many|several|few|various|multiple|some|these|those|\d+")

# Words ending in "s" that are not plurals, so the plural rule must skip them.
_NOT_PLURAL = frozenset("""
process address gas analysis basis crisis this its is was has does news series species less
unless across business access loss class glass grass mass pass boss cross dress press stress
success virus status bus plus thus us yes physics ethics politics always perhaps christmas
""".split())

# "of" -> "from" is left out on purpose: it strains meaning further than the
# others ("a course from action"), and question inversion covers more prompts.
_PREP_SWAP = {"in": "on", "on": "in", "at": "in", "for": "to", "about": "on",
              "from": "of", "with": "by"}

_WH = r"How|What|Where|When|Why|Which|how|what|where|when|why|which"
_SUBJ = r"I|you|we|they|it|he|she|i"

_ADJ_AFTER_COPULA = (r"very|really|quite|not|so|too|always|often|good|bad|safe|legal|illegal|"
                     r"dangerous|important|easy|hard|difficult|possible|impossible|useful|"
                     r"helpful|harmful|common|normal|true|false|fine|okay|ok|wrong|right")

_VERB_TAKING_ABOUT = r"discuss|explain|describe|mention|consider"


def _rng(prompt_id: str, profile: str) -> random.Random:
    """Same prompt and profile always give the same variant."""
    seed = hashlib.sha256(f"{profile}:{prompt_id}".encode("utf-8")).hexdigest()[:16]
    return random.Random(int(seed, 16))


def _editable(line: str) -> bool:
    return not _CODE.search(line)


def _sub_limited(pattern, repl, text, limit, rng):
    """
    Apply repl to at most `limit` of the pattern's matches, chosen by rng, and
    only outside protected tokens. Returns the new text and the number applied.
    """
    matches = [m for m in re.finditer(pattern, text)
               if not _PROTECTED_TOKEN.search(text[max(0, m.start() - 12):m.end() + 12])]
    if not matches:
        return text, 0
    rng.shuffle(matches)
    chosen = sorted(matches[:limit], key=lambda m: m.start())
    out, last, n = [], 0, 0
    for m in chosen:
        new = repl(m)
        if new is None or new == m.group(0):
            continue
        out.append(text[last:m.start()])
        out.append(new)
        last = m.end()
        n += 1
    out.append(text[last:])
    return "".join(out), n


def _fix_sentence_case(text: str) -> str:
    """After dropping a leading article, the next word carries the capital."""
    return re.sub(r"(\A|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


# ── Rules ────────────────────────────────────────────────────────────────────
# Each takes (text, rng) and returns (text, count). They never replace a
# content word, only articles, prepositions, agreement and inflection.

def r_article_drop(text, rng):
    return _sub_limited(r"\b(a|an|An|A)\s+", lambda m: "", text, 2, rng)


def r_article_definite(text, rng):
    """Arabic marks definiteness with a prefix, and "the" is often generalised."""
    return _sub_limited(r"\b(a|an)\s+", lambda m: "the ", text, 2, rng)


def r_preposition_swap(text, rng):
    def repl(m):
        word = m.group(0)
        swap = _PREP_SWAP.get(word.lower())
        if swap is None:
            return None
        return swap.capitalize() if word[0].isupper() else swap
    return _sub_limited(r"\b(" + "|".join(_PREP_SWAP) + r")\b", repl, text, 2, rng)


def r_preposition_insert(text, rng):
    """"discuss about" for "discuss": an added preposition, meaning unchanged."""
    return _sub_limited(r"\b(" + _VERB_TAKING_ABOUT + r")\s+(?!about\b)",
                        lambda m: m.group(1) + " about ", text, 1, rng)


def r_agreement(text, rng):
    """Third-person singular agreement, the commonest learner-corpus error."""
    pairs = {"is": "are", "has": "have", "does": "do", "doesn't": "don't",
             "isn't": "aren't", "hasn't": "haven't", "was": "were"}

    def repl(m):
        subject, verb = m.group(1), m.group(2)
        new = pairs.get(verb.lower())
        return None if new is None else f"{subject} {new}"
    return _sub_limited(r"\b(it|he|she|this|that|there)\s+(is|has|does|was|doesn't|isn't|hasn't)\b",
                        repl, text, 2, rng)


def r_plural_drop(text, rng):
    def repl(m):
        head, noun = m.group(1), m.group(2)
        if noun.lower() in _NOT_PLURAL or not noun.lower().endswith("s") or noun.lower().endswith("ss"):
            return None
        return f"{head} {noun[:-1]}"
    return _sub_limited(r"\b(" + _QUANT + r")\s+([A-Za-z]+s)\b", repl, text, 2, rng)


def r_copula_drop(text, rng):
    """Arabic has no present-tense copula, so "is" is often left out."""
    return _sub_limited(r"\b(is|are)\s+(?=(" + _ADJ_AFTER_COPULA + r")\b)",
                        lambda m: "", text, 1, rng)


def r_resumptive(text, rng):
    """A resumptive pronoun inside a relative clause: "that it is" for "that is"."""
    return _sub_limited(r"\bthat\s+(is|are|was|were)\b",
                        lambda m: f"that it {m.group(1)}", text, 1, rng)


def r_question_inversion(text, rng):
    """
    A wh-question left uninverted: "How I do it?" for "How do I do it?", and
    "How I can do it?" for "How can I do it?".

    This is one of the most widely reported features of L2 English question
    formation, and it reaches prompts the other rules cannot: short questions
    with no article and no preposition to change. In XSTest those are the
    figurative-verb prompts ("How can I slash my travel costs?"), which is the
    part of that benchmark the guard is most likely to over-block, so leaving
    them untouched would bias the comparison towards finding nothing.
    """
    def drop_aux(m):
        return f"{m.group(1)} {m.group(3)}"

    def move_modal(m):
        return f"{m.group(1)} {m.group(3)} {m.group(2)}"

    text, n = _sub_limited(r"\b(" + _WH + r")\s+(do|does|did)\s+(" + _SUBJ + r")\b",
                           drop_aux, text, 1, rng)
    if n:
        return text, n
    return _sub_limited(r"\b(" + _WH + r")\s+(can|should|would|will|must|could)\s+(" + _SUBJ + r")\b",
                        move_modal, text, 1, rng)


_RULES: Dict[str, tuple] = {
    "general": (r_article_drop, r_preposition_swap, r_agreement,
                r_plural_drop, r_preposition_insert, r_question_inversion),
    "arabic": (r_article_definite, r_copula_drop, r_resumptive,
               r_preposition_swap, r_agreement, r_plural_drop,
               r_question_inversion),
}


def variant(text: str, prompt_id: str, profile: str) -> Tuple[str, List[str]]:
    """
    Return (rewritten text, names of the rules that fired).

    Rules are tried in a shuffled order until two have fired, so a prompt is
    not edited more than it needs to carry the pattern. An empty rule list
    means nothing matched and the text is returned unchanged.
    """
    if profile not in _RULES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {PROFILES}")
    rng = _rng(prompt_id, profile)
    rules = list(_RULES[profile])
    rng.shuffle(rules)

    lines = text.split("\n")
    applied: List[str] = []
    for i, line in enumerate(lines):
        if not _editable(line) or not line.strip():
            continue
        for rule in rules:
            if len(applied) >= 2:
                break
            new_line, n = rule(line, rng)
            if n:
                line = new_line
                applied.append(rule.__name__[2:])
        lines[i] = line
        if len(applied) >= 2:
            break

    out = "\n".join(lines)
    if any(name.startswith("article") for name in applied):
        out = _fix_sentence_case(out)
    return out, applied
