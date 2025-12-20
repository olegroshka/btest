import numpy as np

from quantdsl_backtest.models.volume_limits import VolumeLimitModel, build_volume_limit_model


def test_cap_notional_change_clip_behavior():
    m = VolumeLimitModel(max_participation=0.1, mode="clip", min_fill_notional=0.0)
    price = 10.0
    volume = 1000.0
    max_notional = 0.1 * volume * price  # 1000

    # Below cap stays unchanged
    assert abs(m.cap_notional_change(dn=500.0, price=price, volume=volume) - 500.0) < 1e-12

    # Above cap is clipped to +/- max_notional
    assert abs(m.cap_notional_change(dn=2000.0, price=price, volume=volume) - max_notional) < 1e-12
    assert abs(m.cap_notional_change(dn=-2500.0, price=price, volume=volume) + max_notional) < 1e-12

    # Unlimited when mp>=1 -> unchanged
    m2 = VolumeLimitModel(max_participation=1.0, mode="clip", min_fill_notional=0.0)
    assert abs(m2.cap_notional_change(dn=5000.0, price=price, volume=volume) - 5000.0) < 1e-12


def test_min_fill_notional():
    m = VolumeLimitModel(max_participation=1.0, mode="clip", min_fill_notional=100.0)
    assert not m.passes_min_fill(99.99)
    assert m.passes_min_fill(100.0)


def test_build_volume_limit_model_factory():
    class DummyCfg:
        max_participation = 0.2
        mode = "proportional"
        min_fill_notional = 50.0

    vm = build_volume_limit_model(DummyCfg())
    assert isinstance(vm, VolumeLimitModel)
    assert abs(vm.max_participation - 0.2) < 1e-12
    assert vm.mode == "proportional"
    assert abs(vm.min_fill_notional - 50.0) < 1e-12
