import pytest

from rag.chemicals import (
    ChemicalIdentity,
    cas_is_valid,
    detect_chemicals,
    extract_identity,
    find_cas_numbers,
    make_identity,
)


@pytest.mark.parametrize("cas", ["67-64-1", "71-43-2", "64-17-5", "67-56-1", "7732-18-5"])
def test_valid_cas_numbers(cas):
    assert cas_is_valid(cas)


@pytest.mark.parametrize("cas", ["67-64-2", "10071-43-2", "1-23-4", "abc", "67641"])
def test_invalid_cas_numbers(cas):
    assert not cas_is_valid(cas)


def test_find_cas_ignores_bad_checksums():
    assert find_cas_numbers("CAS 67-64-1 and bogus 67-64-9") == ["67-64-1"]


# Text excerpts copied from the three real SDS layouts in data/.
CARL_ROTH = """Acetone ≥99,5 %, for synthesis
SECTION 1: Identification of the substance/mixture and of the company/undertaking
1.1 Product identifier
Identification of the substance Acetone ≥99,5 %, for synthesis
CAS number 67-64-1
Alternative name(s) 2-Propanone
SECTION 3: Composition/information on ingredients
Name of substance Acetone"""

SIGMA = """SECTION 1: Identification of the substance/mixture and of the company/undertaking
Product name : Methanol
CAS-No. : 67-56-1
Synonyms : Methyl alcohol"""

AIRGAS = """BenzenePhenyl hydride; Benzol; benzene, pure;
Section 1. Identification:::Chemical name:benzene
Product use:Synthetic/Analytical chemistry.Synonym:Phenyl hydride; Benzol; benzene, pure; BENZOL DILUENT;annulene; carbon oilSDS #:001062
benzene 10071-43-2Ingredient nameCAS number%
Chemical name:benzeneOther means of identification:Phenyl hydride; Benzol;CAS number:71-43-2"""


def test_extracts_identity_from_carl_roth_layout():
    ident = extract_identity([CARL_ROTH], "Safety Data Sheet: Acetone", "acetone_sds.pdf")
    assert (ident.key, ident.name, ident.cas) == ("acetone", "Acetone", "67-64-1")
    assert "2-Propanone" in ident.synonyms


def test_extracts_identity_from_sigma_layout():
    ident = extract_identity([SIGMA], "", "methanol_sds.pdf")
    assert (ident.key, ident.name, ident.cas) == ("methanol", "Methanol", "67-56-1")
    assert "methyl alcohol" in ident.aliases


def test_extracts_identity_from_glued_airgas_layout():
    ident = extract_identity([AIRGAS], "Benzene - Airgas - United States (US) SDS", "benzene_sds.pdf")
    assert (ident.key, ident.name, ident.cas) == ("benzene", "Benzene", "71-43-2")
    assert "Benzol" in ident.synonyms
    assert not any("CAS" in s or "SDS" in s for s in ident.synonyms)


def test_falls_back_to_title_then_filename():
    assert extract_identity(["nothing useful"], "Safety Data Sheet: Toluene", "x.pdf").key == "toluene"
    assert extract_identity(["nothing useful"], "", "sodium_chloride_sds.pdf").name == "Sodium chloride"


@pytest.fixture
def identities() -> list[ChemicalIdentity]:
    return [
        make_identity("Acetone", "67-64-1", ["2-Propanone"]),
        make_identity("Ethanol", "64-17-5", ["Ethyl alcohol"]),
        make_identity("Methanol", "67-56-1", ["Methyl alcohol"]),
        make_identity("Benzene", "71-43-2", ["Benzol"]),
    ]


@pytest.mark.parametrize(
    ("question", "present"),
    [
        ("What is the flash point of methanol?", ("methanol",)),
        ("What is the flash point of ethanol?", ("ethanol",)),
        ("Flash point of METHYL ALCOHOL please", ("methanol",)),
        ("CAS number of 2-propanone?", ("acetone",)),
        ("Which chemical has CAS 71-43-2?", ("benzene",)),
        ("Compare acetone and methanol flash points", ("acetone", "methanol")),
        ("methanol vs ethanol vs benzol", ("methanol", "ethanol", "benzene")),
        ("Is alcohol flammable?", ()),  # ambiguous: no guess
        ("What PPE do I need?", ()),
    ],
)
def test_detects_chemicals(identities, question, present):
    assert detect_chemicals(question, identities).present == present


def test_methanol_is_not_misread_as_ethanol(identities):
    m = detect_chemicals("Tell me about methanol storage", identities)
    assert m.present == ("methanol",)


def test_reports_known_chemicals_missing_from_knowledge_base(identities):
    m = detect_chemicals("What is the flash point of toluene?", identities)
    assert m.present == () and m.missing == ("toluene",)
    m = detect_chemicals("Can acetone be stored with hydrogen peroxide?", identities)
    assert m.present == ("acetone",) and m.missing == ("hydrogen peroxide",)


def test_identity_round_trips_through_dict(identities):
    for ident in identities:
        again = ChemicalIdentity.from_dict(ident.to_dict())
        assert again == ident and again.aliases == ident.aliases
