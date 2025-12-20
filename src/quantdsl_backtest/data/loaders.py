from __future__ import annotations

from typing import Optional

from ..dsl.data_config import DataConfig
from ..dsl.universe import Universe
from .requests import DataRequest
from .resolver import DataConfigAdapter
from .schema import MarketData


def load_data_request_as_market_data(
    request: DataRequest,
    data_cfg: DataConfig,
    universe: Optional[Universe],
) -> MarketData:
    """Compatibility loader from DataRequest -> MarketData.

    All currently supported market data schemes are now routed via the provider
    layer (orchestrator + registry).

    Unsupported schemes raise a clear error.
    """

    from .orchestrator import load_bundle
    from .bundles import MarketBarsBundle

    bundle = load_bundle(request, universe)
    if not isinstance(bundle, MarketBarsBundle):
        raise TypeError(
            f"Expected MarketBarsBundle for {request.source!r}, got {type(bundle)}"
        )

    return MarketData(
        bars=bundle.bars,
        instruments=bundle.instruments,
        fields=bundle.fields,
        frequency=bundle.frequency,
        calendar=bundle.calendar or data_cfg.calendar,
        tz=bundle.tz,
    )


def load_data_config_as_market_data(
    data_cfg: DataConfig,
    universe: Optional[Universe],
) -> MarketData:
    """Entry point used by the legacy API: DataConfig -> DataRequest -> MarketData."""

    request = DataConfigAdapter.to_request(data_cfg, kind="market_bars")
    return load_data_request_as_market_data(request, data_cfg, universe)
