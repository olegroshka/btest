from __future__ import annotations

import os
import pathlib
import tempfile

from quantdsl_backtest.examples.lagging_indecies import build_strategy
from quantdsl_backtest.engine.backtest_runner import run_backtest
from quantdsl_backtest.platform_api.services.catalog import default_arctic_client
from quantdsl_backtest.platform_api.services.catalog_meta_refresh import refresh_catalog_meta_from_cache


def main() -> None:
    tmp = pathlib.Path(tempfile.mkdtemp())
    ar = tmp / "arctic"
    ar.mkdir(parents=True, exist_ok=True)
    os.environ["QUANTDSL_ARCTIC_URI"] = f"lmdb://{ar.as_posix()}"

    run_backtest(build_strategy())

    arctic = default_arctic_client()
    print("libraries:", list(arctic.list_libraries()))

    stats = refresh_catalog_meta_from_cache(arctic=arctic)
    print("refresh stats:", stats)

    for lib_name in arctic.list_libraries():
        if not str(lib_name).startswith("market_data/"):
            continue
        lib = arctic.get_library(lib_name)
        syms = list(lib.list_symbols())
        print("\nlib", lib_name, "symbols", len(syms))
        for s in syms:
            try:
                obj = lib.read(s)
                data = getattr(obj, "data", obj)
                print(" ", s, "type=", type(data))
            except Exception as e:
                print(" ", s, "READ_ERROR", repr(e))


if __name__ == "__main__":
    main()

