"""Mode selection criteria: MDL, LZ compressibility, RG relevance (M3.3-T1/T2/T3)."""

from __future__ import annotations

import numpy as np

from quantdsl_backtest.smim.interfaces import ModalFrame


# ─────────────────────────────────────────────────────────────────────────────
# LZ-76 complexity
# ─────────────────────────────────────────────────────────────────────────────

def lz_complexity(sequence: np.ndarray) -> int:
    """Lempel-Ziv 76 complexity — number of phrases in sequential parsing.

    Binarises the sequence (1 if > median, 0 otherwise) then performs
    the LZ76 sequential parsing algorithm.
    """
    seq = np.asarray(sequence, dtype=float)
    if seq.size == 0:
        return 0
    binary = (seq > np.median(seq)).astype(int)
    s = "".join(map(str, binary))
    n = len(s)
    complexity = 1
    # c: current pointer into already-parsed string
    c = 1
    # l: length of current match attempt
    l = 1
    # i: end of current phrase
    i = 0
    while c + l <= n:
        # Check if s[c:c+l] appears in s[0:c+l-1]
        sub = s[c : c + l]
        if sub in s[: c + l - 1]:
            l += 1
        else:
            complexity += 1
            i = c + l
            c = i
            l = 1
    if l > 1:
        # Partial phrase at end — count it
        pass  # already counted in complexity if phrase was started
    return complexity


# ─────────────────────────────────────────────────────────────────────────────
# MDL Mode Selector
# ─────────────────────────────────────────────────────────────────────────────

class MDLModeSelector:
    """MDL criterion for mode selection.

    For each k (number of retained modes), computes a description length:
        DL(k) = T * log(residual_variance(k)) + k * N * log(T) / 2
    Selects k* minimising DL.
    residual_variance(k) = ||Y - U_k U_k^T Y||^2_F / (T * N)

    The penalty term k*N*log(T)/2 scales with both the dimension N and log(T),
    matching the BIC cost of estimating k orthonormal N-vectors from T observations.
    This ensures the criterion favours noise-mode rejection when N is large relative
    to T/N (the fractional variance explained per noise SVD mode).
    """

    def select(
        self,
        modal_frame: ModalFrame,
        time_series: np.ndarray,
    ) -> tuple[ModalFrame, np.ndarray]:
        """Select optimal number of modes via MDL.

        Parameters
        ----------
        modal_frame:
            ModalFrame with K candidate modes.
        time_series:
            (T, N) or (T, K) array. If shape is (T, K), treated as modal amplitudes
            projected onto basis. If (T, N), we use the basis to project.

        Returns
        -------
        (reduced_frame, reduced_time_series) with k* modes.
        """
        ts = np.asarray(time_series, dtype=float)
        U = modal_frame.basis  # (N, K)
        K = U.shape[1]
        N = U.shape[0]

        # Build Y: (T, N) signal matrix
        # If time_series is (T, K) we reconstruct Y = ts @ U.T
        if ts.ndim == 1:
            ts = ts[:, np.newaxis]
        T = ts.shape[0]

        if ts.shape[1] == N:
            Y = ts  # (T, N)
        elif ts.shape[1] == K:
            Y = ts @ U.T  # (T, N)
        else:
            # Best effort: use as-is padded/trimmed
            Y = ts[:, :N] if ts.shape[1] > N else np.pad(ts, ((0, 0), (0, N - ts.shape[1])))

        # Compute DL for k = 1, ..., K
        best_k = 1
        best_dl = float("inf")

        Y_mean = Y.mean(axis=0)  # (N,)
        ss_total = np.sum((Y - Y_mean) ** 2)

        for k in range(1, K + 1):
            U_k = U[:, :k]  # (N, k)
            # Project Y onto span(U_k): Y_hat = Y @ U_k @ U_k.T
            coefs = Y @ U_k  # (T, k)
            Y_hat = coefs @ U_k.T  # (T, N)
            residual = Y - Y_hat
            res_var = np.sum(residual ** 2) / (T * N)
            if res_var <= 0:
                res_var = 1e-300
            dl = T * np.log(res_var) + k * N * np.log(max(T, 2)) / 2
            if dl < best_dl:
                best_dl = dl
                best_k = k

        # Build reduced frame
        reduced_basis = U[:, :best_k]
        reduced_eigs = modal_frame.eigenvalues[:best_k]
        reduced_frame = ModalFrame(
            basis=reduced_basis,
            eigenvalues=reduced_eigs,
            method=modal_frame.method,
            metadata=modal_frame.metadata,
        )
        # Reduced time series: modal amplitudes (T, best_k)
        reduced_ts = Y @ reduced_basis  # (T, best_k)
        return reduced_frame, reduced_ts


# ─────────────────────────────────────────────────────────────────────────────
# Compressibility Filter
# ─────────────────────────────────────────────────────────────────────────────

class CompressibilityFilter:
    """Retain modes whose LZ compressibility rho >= min_compressibility.

    rho_k = 1 - lz_complexity(c_k) * log2(len(c_k)) / len(c_k)

    This is the standard normalisation from Lempel & Ziv (1976): c(s)·log₂(n)/n
    converges to the entropy rate for ergodic sources (Cover & Thomas, Thm 12.10.1).
    Under this formula random sequences give rho ≈ 0 (incompressible) while
    constant/periodic sequences give rho close to 1.
    """

    def __init__(self, min_compressibility: float = 0.1) -> None:
        self.min_compressibility = min_compressibility

    def select(
        self,
        modal_frame: ModalFrame,
        time_series: np.ndarray,
    ) -> tuple[ModalFrame, np.ndarray]:
        ts = np.asarray(time_series, dtype=float)
        if ts.ndim == 1:
            ts = ts[:, np.newaxis]
        K = modal_frame.K
        T = ts.shape[0]

        # If time_series columns == N (actors), project onto modes
        N = modal_frame.N
        if ts.shape[1] == N:
            amplitudes = ts @ modal_frame.basis  # (T, K)
        elif ts.shape[1] == K:
            amplitudes = ts
        else:
            amplitudes = ts[:, :K] if ts.shape[1] >= K else np.pad(ts, ((0, 0), (0, K - ts.shape[1])))

        retained = []
        for k in range(K):
            c_k = amplitudes[:, k]
            lz = lz_complexity(c_k)
            n = max(len(c_k), 2)
            rho = float(np.clip(1.0 - lz * np.log2(n) / n, 0.0, 1.0))
            if rho >= self.min_compressibility:
                retained.append(k)

        if not retained:
            # Keep at least the first mode
            retained = [0]

        idx = np.array(retained, dtype=int)
        reduced_frame = ModalFrame(
            basis=modal_frame.basis[:, idx],
            eigenvalues=modal_frame.eigenvalues[idx],
            method=modal_frame.method,
            metadata=modal_frame.metadata,
        )
        reduced_ts = amplitudes[:, idx]
        return reduced_frame, reduced_ts


# ─────────────────────────────────────────────────────────────────────────────
# RG Relevance Filter
# ─────────────────────────────────────────────────────────────────────────────

class RGRelevanceFilter:
    """Coarse-graining stability filter.

    For each mode, checks whether the mode persists at coarser scales
    (lower layer numbers) using layer-averaged loadings.
    """

    def select(
        self,
        modal_frame: ModalFrame,
        time_series: np.ndarray,
        layer_assignments: np.ndarray | None = None,
    ) -> tuple[ModalFrame, np.ndarray]:
        """Filter modes by RG relevance.

        Parameters
        ----------
        modal_frame:
            ModalFrame to filter.
        time_series:
            (T, K) or (T, N) modal amplitudes.
        layer_assignments:
            (N,) integer array assigning each actor to layer 0-3.
            If None, all modes pass.
        """
        ts = np.asarray(time_series, dtype=float)
        if ts.ndim == 1:
            ts = ts[:, np.newaxis]
        K = modal_frame.K
        N = modal_frame.N

        if ts.shape[1] == N:
            amplitudes = ts @ modal_frame.basis  # (T, K)
        elif ts.shape[1] == K:
            amplitudes = ts
        else:
            amplitudes = ts[:, :K] if ts.shape[1] >= K else np.pad(ts, ((0, 0), (0, K - ts.shape[1])))

        if layer_assignments is None:
            # No layer info: all modes pass
            return modal_frame, amplitudes

        layers = np.asarray(layer_assignments, dtype=int)
        unique_layers = np.unique(layers)
        n_layers = len(unique_layers)

        retained = []
        for k in range(K):
            loadings = np.abs(modal_frame.basis[:, k])  # (N,)
            if n_layers <= 1:
                retained.append(k)
                continue
            # Compute mean loading per layer
            layer_means = {}
            layer_sizes = {}
            for l in unique_layers:
                mask = layers == l
                layer_means[l] = loadings[mask].mean() if mask.any() else 0.0
                layer_sizes[l] = mask.sum()
            # Coarse-grained contributions: g_k(l) = mu_l * size_l
            cg = {l: layer_means[l] * layer_sizes[l] for l in unique_layers}

            # Mode is RG-relevant if it has influence at coarse (low-index) layers,
            # i.e. not concentrated ONLY in layer 3.
            # Simple heuristic: pass if max loading layer is NOT exclusively layer 3.
            max_cg_layer = max(cg, key=lambda l: cg[l])
            if max_cg_layer == 3 and all(cg[l] < 1e-12 for l in unique_layers if l != 3):
                # Concentrated only in layer 3 -> fails RG
                pass
            else:
                retained.append(k)

        if not retained:
            retained = [0]

        idx = np.array(retained, dtype=int)
        reduced_frame = ModalFrame(
            basis=modal_frame.basis[:, idx],
            eigenvalues=modal_frame.eigenvalues[idx],
            method=modal_frame.method,
            metadata=modal_frame.metadata,
        )
        return reduced_frame, amplitudes[:, idx]


# ─────────────────────────────────────────────────────────────────────────────
# Combined Mode Selector
# ─────────────────────────────────────────────────────────────────────────────

class CombinedModeSelector:
    """Applies MDL + compressibility + RG relevance in sequence."""

    def __init__(
        self,
        mdl: bool = True,
        min_compressibility: float = 0.1,
        rg_relevance: bool = True,
        layer_assignments: np.ndarray | None = None,
    ) -> None:
        self.mdl = mdl
        self.min_compressibility = min_compressibility
        self.rg_relevance = rg_relevance
        self.layer_assignments = layer_assignments

        self._mdl_selector = MDLModeSelector() if mdl else None
        self._comp_filter = CompressibilityFilter(min_compressibility) if min_compressibility > 0 else None
        self._rg_filter = RGRelevanceFilter() if rg_relevance else None

    def select(
        self,
        modal_frame: ModalFrame,
        time_series: np.ndarray,
    ) -> tuple[ModalFrame, np.ndarray]:
        frame = modal_frame
        ts = np.asarray(time_series, dtype=float)
        if ts.ndim == 1:
            ts = ts[:, np.newaxis]

        # Project to (T, K) modal amplitudes if needed
        N = frame.N
        K = frame.K
        if ts.shape[1] == N:
            ts = ts @ frame.basis  # (T, K)

        if self._mdl_selector is not None:
            frame, ts = self._mdl_selector.select(frame, ts)

        if self._comp_filter is not None:
            frame, ts = self._comp_filter.select(frame, ts)

        if self._rg_filter is not None:
            frame, ts = self._rg_filter.select(frame, ts, self.layer_assignments)

        # Ensure at least 1 mode retained
        if frame.K == 0:
            frame = ModalFrame(
                basis=modal_frame.basis[:, :1],
                eigenvalues=modal_frame.eigenvalues[:1],
                method=modal_frame.method,
                metadata=modal_frame.metadata,
            )
            ts_orig = np.asarray(time_series, dtype=float)
            if ts_orig.ndim == 1:
                ts_orig = ts_orig[:, np.newaxis]
            if ts_orig.shape[1] == N:
                ts = ts_orig @ frame.basis
            else:
                ts = ts_orig[:, :1]

        return frame, ts
