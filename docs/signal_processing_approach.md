# The 4th Way: Signal Processing in Algorithmic Trading

**Source:** [Reddit r/algotrading post by u/if-not-null](https://www.reddit.com/r/algotrading/comments/hyvzq1/the_4th_way_of_algorithmic_trading_signal/)

---

## Overview

The author (EE engineer, 20+ years software dev, 5+ years trading) proposes Signal Processing as a 4th paradigm for algo trading, alongside:

1. Technical Analysis
2. Statistics and Probability  
3. Machine Learning

Renaissance Technologies reportedly uses similar approaches: [Financial Signal Processing (Wikipedia)](https://en.wikipedia.org/wiki/Financial_signal_processing)

---

## Core Concepts That Inspired the Approach

The author explicitly mentions these as key inspirations:

| Concept | Wikipedia Link | Likely Application |
|---------|----------------|-------------------|
| **Information Theory** | [Link](https://en.wikipedia.org/wiki/Information_theory) | Quantifying "surprise" or information content in price moves |
| **Signal Processing** | [Link](https://en.wikipedia.org/wiki/Signal_processing) | Transforming raw price into actionable signals |
| **ADC (Analog-to-Digital Converter)** | [Link](https://en.wikipedia.org/wiki/Analog-to-digital_converter) | Discretizing continuous price streams |

---

## What He's Likely Doing (Inference from Clues)

### Key Characteristics of His Method
- **Parameterless** — no optimization, adapts intrinsically
- **Fast and "natural"** — computationally efficient
- **Transforms price data into "meaningful, tradeable form"**
- **NOT noise filtering** — "I don't think there is noise in price movements"
- **Works across assets and timeframes** (cross-validated)

### Most Probable Signal Processing Techniques

#### 1. **ADC-Inspired Quantization / Event-Based Sampling**
The ADC reference is the biggest hint. Instead of sampling at fixed time intervals:
- **Delta Encoding**: Measure price changes, not absolute prices
- **Threshold Crossing Detection**: Generate signals when price moves by X% (like a delta-sigma ADC)
- **Run-Length Encoding**: Compress consecutive same-direction moves
- **Tick Bars / Volume Bars / Dollar Bars**: Non-uniform sampling based on activity

```python
# Conceptual: event-based signal (not time-based)
def delta_signal(prices, threshold):
    signals = []
    anchor = prices[0]
    for p in prices:
        if abs(p - anchor) / anchor >= threshold:
            signals.append(np.sign(p - anchor))
            anchor = p
    return signals
```

#### 2. **Information-Theoretic Measures**
- **Entropy of returns**: Measure "surprise" in price moves
- **Mutual Information**: Correlation between past and future moves
- **Kolmogorov Complexity**: Compressibility of price sequences
- **Shannon Entropy on discretized returns**: Identify regime changes

```python
# Example: entropy-based regime detection
from scipy.stats import entropy
def rolling_entropy(returns, window=20, bins=10):
    hist, _ = np.histogram(returns[-window:], bins=bins, density=True)
    return entropy(hist + 1e-10)
```

#### 3. **Zero-Crossing / Sign Analysis**
Treating price as a signal and analyzing:
- Zero-crossings of momentum
- Sign changes in returns
- Duration between crossings

#### 4. **Hilbert Transform / Instantaneous Phase**
Extract instantaneous frequency/phase of price oscillations:
```python
from scipy.signal import hilbert
analytic_signal = hilbert(price_series)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_freq = np.diff(instantaneous_phase)
```

#### 5. **Adaptive Filtering Without Parameters**
- **LMS/RLS filters** that self-tune
- **Kalman Filter** (mentioned in comments as related)
- **Empirical Mode Decomposition (EMD)**: Data-driven, no predefined basis

---

## What He Says It's NOT

From the comments:
- **Not FFT-based** (he deflected this question)
- **Not noise filtering** ("I don't think there is noise in price movements")
- **Not traditional indicator optimization**

---

## The "Proverbs" Layer (Decision Rules)

He layered wisdom-based decision rules on top of the signal processing:

| Proverb | Possible Implementation |
|---------|------------------------|
| "The bamboo that bends is stronger than the oak that resists" | Adaptive position sizing, yield to strong trends |
| "When rainwater rises... wait until it settles" (Sun-Tzu) | Don't trade during volatility spikes, wait for mean reversion |
| "If you do not expect the unexpected, you will not find it" (Heraclitus) | Outlier detection, prepare for regime changes |

These are essentially **parameterless regime-aware rules**.

---

## Practical Implementation Ideas for Your DSL

### 1. Event-Based Signal Generation
```python
# Instead of: signal = close.rolling(20).mean() > close.rolling(50).mean()
# Try: signal based on cumulative normalized price movements

def adaptive_momentum(prices, lookback=None):
    """Parameterless momentum using cumsum of sign(returns)"""
    returns = prices.pct_change()
    # Count up-moves vs down-moves adaptively
    cum_sign = np.cumsum(np.sign(returns))
    # Normalize by sqrt(n) for statistical meaning
    n = np.arange(1, len(returns) + 1)
    normalized = cum_sign / np.sqrt(n)
    return normalized
```

### 2. Information Content Signal
```python
def price_surprise(prices, window=20):
    """High surprise = potential reversal, low surprise = trend continuation"""
    returns = prices.pct_change()
    rolling_std = returns.rolling(window).std()
    surprise = abs(returns) / (rolling_std + 1e-10)
    return surprise
```

### 3. ADC-Style Discretization
```python
def discretize_price_moves(prices, levels=5):
    """Convert continuous prices to discrete levels (like ADC)"""
    returns = prices.pct_change()
    # Adaptive quantization based on recent volatility
    vol = returns.rolling(20).std()
    quantized = np.round(returns / vol).clip(-levels, levels)
    return quantized
```

---

## Key Takeaways

1. **Think like an EE**: Price is a signal, not a number
2. **Parameterless is key**: The technique adapts without tuning
3. **ADC is the core hint**: Event-based, threshold-based, or quantized signals
4. **Information theory**: Measure surprise/entropy, not just price level
5. **Proverbs = heuristic rules**: Human wisdom encoded as regime filters

---

## Further Reading

- [Financial Signal Processing (Wikipedia)](https://en.wikipedia.org/wiki/Financial_signal_processing)
- Kalman Filters for pairs trading
- Empirical Mode Decomposition (EMD) for non-stationary signals
- Tick/Volume/Dollar bars (Marcos López de Prado - "Advances in Financial ML")
- Information-theoretic approaches to market microstructure

---

## Comments Worth Noting

- **Kalman filters** mentioned as "bread and butter" for pairs trading
- **DSP is brittle** for direct trading but excellent for **feature engineering**
- Many early Kaggle winners were hardware/firmware engineers using DSP techniques
