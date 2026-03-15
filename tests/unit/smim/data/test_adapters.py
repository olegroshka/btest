"""Stub tests for smim/data/ adapters (M1.2-T2 through M1.2-T7).

Each test class will be filled in as the corresponding adapter is implemented.
Until then, every test is marked xfail so the suite stays green.
"""

import pytest


@pytest.mark.xfail(reason="FredAdapter not yet implemented (M1.2-T2)", strict=False)
class TestFredAdapter:
    def test_fetch_returns_dataframe(self) -> None:
        pytest.fail("not implemented")

    def test_vintage_retrieval_respects_as_of(self) -> None:
        pytest.fail("not implemented")

    def test_source_name(self) -> None:
        pytest.fail("not implemented")


@pytest.mark.xfail(reason="EdgarAdapter not yet implemented (M1.2-T3)", strict=False)
class TestEdgarAdapter:
    def test_fetch_xbrl_tags(self) -> None:
        pytest.fail("not implemented")

    def test_source_name(self) -> None:
        pytest.fail("not implemented")


@pytest.mark.xfail(reason="GdeltAdapter not yet implemented (M1.2-T4)", strict=False)
class TestGdeltAdapter:
    def test_fetch_theme_intensity(self) -> None:
        pytest.fail("not implemented")

    def test_source_name(self) -> None:
        pytest.fail("not implemented")


@pytest.mark.xfail(reason="PitStore not yet implemented (M1.3-T2)", strict=False)
class TestPitStore:
    def test_write_and_read_roundtrip(self) -> None:
        pytest.fail("not implemented")

    def test_pub_date_filtering_prevents_lookahead(self) -> None:
        pytest.fail("not implemented")
