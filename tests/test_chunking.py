from rag.chemicals import make_identity
from rag.chunking import Chunk, chunk_document, remove_boilerplate, segment_sections
from tests.helpers import sds_pages

ACETONE = make_identity("Acetone", "67-64-1")


def test_boilerplate_removed_but_repeated_content_kept():
    pages = [f"ACME Corp SDS\nPage {i} / 5\nSECTION {i}: x\nNot available.\nbody {i}\nFooter v1" for i in range(1, 6)]
    cleaned = remove_boilerplate(pages, head=2, tail=1)
    for i, page in enumerate(cleaned, start=1):
        assert "ACME Corp SDS" not in page
        assert f"Page {i}" not in page
        assert "Footer v1" not in page
        assert "Not available." in page  # mid-page repeated content survives
        assert f"body {i}" in page


def test_segments_ignore_cross_references_and_backward_references():
    pages = [
        "SECTION 1: Identification\nProduct\nSECTION 2: Hazards identification\nflammable",
        "SECTION 7: Handling and storage\nIncompatible materials: see section 10. Keep away from heat.\n"
        "Refer to Section 2: Hazards identification of this SDS.\nSECTION 8: Exposure controls\ngloves",
    ]
    nums = [n for n, _ in segment_sections(pages)]
    assert nums == [1, 2, 7, 8]


def test_segments_split_headings_glued_mid_line():
    pages = ["intro text 1-866-734-3438Section 2. Hazards identificationFLAMMABLE LIQUIDS"
             "\nmore Section 3. Composition/information on ingredients benzene"]
    segs = segment_sections(pages)
    assert [n for n, _ in segs] == [0, 2, 3]
    assert segs[0][1][0][0].strip() == "intro text 1-866-734-3438"
    assert segs[1][1][0][0].startswith("Section 2. Hazards identificationFLAMMABLE")


def test_running_section_headers_continue_section():
    pages = ["Section 4. First aid measures\neye: rinse", "BenzeneSection 4. First aid measures\nskin: wash",
             "Section 5. Fire-fighting measures\nfoam"]
    segs = segment_sections(pages)
    assert [n for n, _ in segs] == [4, 5]
    assert [p for _, p in segs[0][1]] == [1, 1, 2, 2]


def _chunks(chunk_size=500, overlap=60) -> list[Chunk]:
    pages = sds_pages("Acetone", "67-64-1", "-17 C", extra={9: [f"Property {i}: value {i}" for i in range(40)]})
    return chunk_document(["\n".join(p) for p in pages], ACETONE, "acetone.pdf", "a" * 64, chunk_size, overlap)


def test_chunks_carry_metadata_and_page_ranges():
    chunks = _chunks()
    flash = [c for c in chunks if "Flash point -17 C" in c.text]
    assert len(flash) >= 1
    c = flash[0]
    assert (c.section, c.page_start, c.chemical_key, c.source) == (9, 3, "acetone", "acetone.pdf")
    assert c.header == "Acetone | Section 9: Physical and chemical properties"
    assert c.search_text.startswith(c.header)
    assert len({ch.chunk_id for ch in chunks}) == len(chunks)


def test_chunks_respect_size_and_never_mix_sections():
    chunks = _chunks(chunk_size=300, overlap=40)
    assert all(len(c.text) <= 300 for c in chunks)
    s9 = [c for c in chunks if c.section == 9]
    assert len(s9) > 1, "long section should be subdivided"
    assert all("SECTION 10" not in c.text for c in s9)


def test_overlap_repeats_trailing_lines():
    s9 = [c for c in _chunks(chunk_size=300, overlap=60) if c.section == 9]
    first_last_line = s9[0].text.splitlines()[-1]
    assert first_last_line in s9[1].text


def test_short_no_data_sections_are_kept_as_evidence():
    chunks = _chunks()
    eco = [c for c in chunks if c.section == 12]
    assert eco and "No data available." in eco[0].text


def test_chunk_round_trip():
    c = _chunks()[0]
    assert Chunk.from_dict(c.to_dict()) == c
