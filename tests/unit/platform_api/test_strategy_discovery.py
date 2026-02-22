from __future__ import annotations


def test_bootstrap_strategies_copies_examples_when_empty(tmp_path):
    from quantdsl_backtest.platform_api.services.strategy_discovery import bootstrap_strategies

    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "a.py").write_text('"""A title\n\nDesc"""\n\nX=1\n', encoding="utf-8")
    (examples / "b.py").write_text('"""B title"""\n\nY=2\n', encoding="utf-8")

    target = tmp_path / "strategies"
    did = bootstrap_strategies(target_dir=target, src_examples_dir=examples)
    assert did is True

    assert (target / "a.py").exists()
    assert (target / "b.py").exists()

    # Idempotent when not empty
    did2 = bootstrap_strategies(target_dir=target, src_examples_dir=examples)
    assert did2 is False


def test_discover_strategies_hash_and_docstring(tmp_path):
    from quantdsl_backtest.platform_api.services.strategy_discovery import discover_strategies

    d = tmp_path / "strategies"
    d.mkdir()

    (d / "s1.py").write_text('"""My Strategy\n\nLonger description here."""\n\nprint("hi")\n', encoding="utf-8")
    (d / "s2.py").write_text("# no docstring\nX=1\n", encoding="utf-8")

    infos = discover_strategies(strategies_dir=d)
    assert [i.id for i in infos] == ["s1", "s2"]

    s1 = infos[0]
    assert s1.name == "My Strategy"
    assert s1.description == "Longer description here."
    assert len(s1.strategy_hash) == 64

    s2 = infos[1]
    assert s2.name is None
    assert s2.description is None
    assert len(s2.strategy_hash) == 64


def test_read_and_write_strategy_source(tmp_path):
    from quantdsl_backtest.platform_api.services.strategy_discovery import read_strategy_source, write_strategy_source

    d = tmp_path / "strategies"
    p = write_strategy_source(strategies_dir=d, strategy_id="My New Strategy", source="print('x')\n")
    assert p.exists()
    assert p.name == "My_New_Strategy.py"

    s = read_strategy_source(strategies_dir=d, strategy_id="My_New_Strategy")
    assert "print('x')" in s
