"""Chemical identity: extraction from SDS text and detection in questions.

Every indexed document is tagged with a chemical identity (name, CAS number,
synonyms). Questions are scanned for those names, aliases and CAS numbers so
retrieval can be restricted to the right SDS and facts from one chemical are
never presented as another's.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

CAS_RE = re.compile(r"(?<![\d-])(\d{2,7}-\d{2}-\d)(?![\d-])")
CAS_LABELLED_RE = re.compile(r"CAS[\s\-]*(?:No\.?|number|#|Reg\.?\s*No\.?)?\s*[:.]?\s*(\d{2,7}-\d{2}-\d)", re.I)

# Common laboratory chemicals: canonical key -> (CAS, aliases). Used to enrich
# identities found in documents and to recognise questions about chemicals
# that are *not* in the knowledge base. Bare "alcohol" is deliberately absent
# because it is ambiguous between ethanol and methanol.
KNOWN_CHEMICALS: dict[str, tuple[str, tuple[str, ...]]] = {
    "acetone": ("67-64-1", ("acetone", "2-propanone", "propan-2-one", "propanone", "dimethyl ketone")),
    "ethanol": ("64-17-5", ("ethanol", "ethyl alcohol", "grain alcohol")),
    "methanol": ("67-56-1", ("methanol", "methyl alcohol", "wood alcohol", "carbinol")),
    "benzene": ("71-43-2", ("benzene", "benzol", "phenyl hydride")),
    "toluene": ("108-88-3", ("toluene", "methylbenzene", "toluol")),
    "xylene": ("1330-20-7", ("xylene", "xylenes", "dimethylbenzene")),
    "isopropanol": ("67-63-0", ("isopropanol", "isopropyl alcohol", "2-propanol", "propan-2-ol", "ipa")),
    "hexane": ("110-54-3", ("hexane", "n-hexane")),
    "chloroform": ("67-66-3", ("chloroform", "trichloromethane")),
    "dichloromethane": ("75-09-2", ("dichloromethane", "methylene chloride", "dcm")),
    "acetonitrile": ("75-05-8", ("acetonitrile", "methyl cyanide")),
    "diethyl ether": ("60-29-7", ("diethyl ether", "ethyl ether")),
    "formaldehyde": ("50-00-0", ("formaldehyde", "formalin")),
    "acetic acid": ("64-19-7", ("acetic acid", "glacial acetic acid")),
    "hydrochloric acid": ("7647-01-0", ("hydrochloric acid", "hydrogen chloride")),
    "sulfuric acid": ("7664-93-9", ("sulfuric acid", "sulphuric acid")),
    "nitric acid": ("7697-37-2", ("nitric acid",)),
    "sodium hydroxide": ("1310-73-2", ("sodium hydroxide", "caustic soda")),
    "ammonia": ("7664-41-7", ("ammonia",)),
    "hydrogen peroxide": ("7722-84-1", ("hydrogen peroxide",)),
}
_CAS_TO_KNOWN = {cas: key for key, (cas, _) in KNOWN_CHEMICALS.items()}

_NAME_PATTERNS = [
    re.compile(r"Name of substance\s*:?\s*([^\n]+)", re.I),
    re.compile(r"Chemical name\s*:+\s*([^\n]+)", re.I),
    re.compile(r"Product name\s*:?\s*([^\n]+)", re.I),
    re.compile(r"Substance name\s*:?\s*([^\n]+)", re.I),
    re.compile(r"Identification of the substance\s*:?\s*([^\n]+)", re.I),
]
_SYNONYM_PATTERNS = [
    re.compile(r"Alternative name\(s\)\s*:?\s*([^\n]+)", re.I),
    re.compile(r"Synonyms?\s*:\s*([^\n]+)", re.I),
    re.compile(r"Other means of identification\s*:\s*([^\n]+)", re.I),
]
_GENERIC_WORDS = {"pure", "crude", "unrefined", "solution", "mixture", "substance", "product", "liquid"}


@dataclass(frozen=True)
class ChemicalIdentity:
    key: str  # canonical lowercase key, e.g. "acetone"
    name: str  # display name, e.g. "Acetone"
    cas: str = ""
    synonyms: tuple[str, ...] = ()
    aliases: tuple[str, ...] = field(default=(), compare=False)  # lowercase match terms

    def to_dict(self) -> dict[str, object]:
        return {"key": self.key, "name": self.name, "cas": self.cas, "synonyms": list(self.synonyms)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChemicalIdentity:
        syn = tuple(str(s) for s in d.get("synonyms") or [])
        return make_identity(str(d["name"]), str(d.get("cas") or ""), syn)


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return text.replace("−", "-").replace("–", "-")


def cas_is_valid(cas: str) -> bool:
    """Validate a CAS Registry Number with its check digit."""
    m = re.fullmatch(r"(\d{2,7})-(\d{2})-(\d)", cas)
    if not m:
        return False
    digits = (m.group(1) + m.group(2))[::-1]
    total = sum((i + 1) * int(d) for i, d in enumerate(digits))
    return total % 10 == int(m.group(3))


def find_cas_numbers(text: str) -> list[str]:
    return [c for c in CAS_RE.findall(text) if cas_is_valid(c)]


def _clean_name(raw: str) -> str:
    s = raw.strip()
    # Split glued words such as "benzeneOther means of identification".
    s = re.split(r"(?<=[a-z])(?=[A-Z][a-z])", s)[0]
    for sep in ("≥", ">=", "%", ",", "(", " - ", ";", ":", "  "):
        s = s.split(sep)[0]
    s = re.sub(r"[\s\d.]+$", "", s).strip(" .-")
    return s


def _plausible_name(s: str) -> bool:
    return 2 <= len(s) <= 60 and bool(re.search(r"[A-Za-z]{2}", s)) and s.lower() not in _GENERIC_WORDS


def _clean_synonyms(raw: str) -> list[str]:
    raw = re.split(r"SDS\s*#|Product code|Product type|CAS", raw)[0]
    out = []
    for part in re.split(r"[;]", raw):
        s = part.strip(" .:")
        if not s or "(" in s or len(s) < 4 or len(s) > 40:
            continue
        low = s.lower()
        if any(w in low.split() for w in _GENERIC_WORDS) or "," in s:
            continue
        out.append(s)
    return out


def make_identity(name: str, cas: str = "", synonyms: Iterable[str] = ()) -> ChemicalIdentity:
    """Build an identity, merging in aliases from ``KNOWN_CHEMICALS``."""
    key = normalise(name).strip()
    known_key = _CAS_TO_KNOWN.get(cas) if cas else None
    if known_key is None and key in KNOWN_CHEMICALS:
        known_key = key
    aliases: set[str] = {key}
    if known_key:
        key = known_key
        known_cas, known_aliases = KNOWN_CHEMICALS[known_key]
        cas = cas or known_cas
        aliases.update(known_aliases)
    syn = tuple(dict.fromkeys(s for s in synonyms if normalise(s) != key))
    aliases.update(normalise(s) for s in syn)
    if cas:
        aliases.add(cas)
    display = name.strip()
    display = display[:1].upper() + display[1:] if display.islower() else display
    return ChemicalIdentity(
        key=key,
        name=display,
        cas=cas,
        synonyms=syn,
        aliases=tuple(sorted(a for a in aliases if len(a) >= 3)),
    )


def extract_identity(pages: list[str], title: str, filename: str) -> ChemicalIdentity:
    """Infer the chemical an SDS describes from its first pages."""
    head = "\n".join(pages[:4])

    name = ""
    for pat in _NAME_PATTERNS:
        for m in pat.finditer(head):
            cand = _clean_name(m.group(1))
            if _plausible_name(cand):
                name = cand
                break
        if name:
            break
    if not name and title:
        t = re.sub(r"(?i)^\s*safety\s+data\s+sheet\s*[:\-]?\s*", "", title)
        cand = _clean_name(t)
        if _plausible_name(cand):
            name = cand
    if not name:
        stem = re.sub(r"(?i)[_\-\s]*(m?sds|safety[_\-\s]*data[_\-\s]*sheet)[_\-\s]*", " ", filename.rsplit(".", 1)[0])
        name = re.sub(r"[_\-]+", " ", stem).strip() or filename

    labelled = [c for c in CAS_LABELLED_RE.findall(head) if cas_is_valid(c)]
    pool = labelled or find_cas_numbers(head)
    cas = Counter(pool).most_common(1)[0][0] if pool else ""

    synonyms: list[str] = []
    for pat in _SYNONYM_PATTERNS:
        for m in pat.finditer(head):
            synonyms.extend(_clean_synonyms(m.group(1)))

    return make_identity(name, cas, synonyms)


def _alias_regex(alias: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])")


@dataclass(frozen=True)
class ChemicalMention:
    present: tuple[str, ...]  # keys of indexed chemicals mentioned, in order of appearance
    missing: tuple[str, ...]  # well-known chemicals mentioned but not indexed


def detect_chemicals(question: str, identities: Iterable[ChemicalIdentity]) -> ChemicalMention:
    """Find which chemicals a question is about.

    Longer aliases are matched first and their spans masked, so
    "methyl alcohol" is not also read as some other chemical's alias.
    """
    text = normalise(question)
    idents = list(identities)
    indexed_keys = {i.key for i in idents}
    terms: list[tuple[str, str, bool]] = []  # (alias, key, indexed)
    for ident in idents:
        terms.extend((a, ident.key, True) for a in ident.aliases)
    for key, (cas, aliases) in KNOWN_CHEMICALS.items():
        if key not in indexed_keys:
            terms.extend((a, key, False) for a in (*aliases, cas))
    terms.sort(key=lambda t: len(t[0]), reverse=True)

    hits: list[tuple[int, str, bool]] = []
    masked = text
    for alias, key, indexed in terms:
        for m in _alias_regex(alias).finditer(masked):
            hits.append((m.start(), key, indexed))
            masked = masked[: m.start()] + " " * (m.end() - m.start()) + masked[m.end() :]
    hits.sort()
    present = tuple(dict.fromkeys(k for _, k, idx in hits if idx))
    missing = tuple(dict.fromkeys(k for _, k, idx in hits if not idx and k not in present))
    return ChemicalMention(present=present, missing=missing)
