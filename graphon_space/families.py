"""Graphon family constructors and optimizer parameterizations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .graphon import StepGraphon


def _rng(seed: int | np.random.Generator | None = None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def upper_triangle_count(k: int) -> int:
    return k * (k + 1) // 2


def upper_triangle_values(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    idx = np.triu_indices(matrix.shape[0])
    return np.asarray(matrix[idx], dtype=float)


def symmetric_from_upper(k: int, values: Sequence[float]) -> NDArray[np.float64]:
    values = np.asarray(values, dtype=float)
    if values.size != upper_triangle_count(k):
        raise ValueError(f"expected {upper_triangle_count(k)} upper-triangle values")
    matrix = np.zeros((k, k), dtype=float)
    idx = np.triu_indices(k)
    matrix[idx] = values
    matrix[(idx[1], idx[0])] = values
    return matrix


def erdos_renyi(p: float, **metadata: Any) -> StepGraphon:
    p = float(np.clip(p, 0.0, 1.0))
    return StepGraphon([1.0], [[p]], {"family": "erdos-renyi", **metadata})


def symmetric_bipodal(a: float, d: float, **metadata: Any) -> StepGraphon:
    return StepGraphon(
        [0.5, 0.5],
        [[float(a), float(d)], [float(d), float(a)]],
        {"family": "symmetric-bipodal", **metadata},
    )


def bipodal(c: float, a: float, b: float, d: float, **metadata: Any) -> StepGraphon:
    c = float(np.clip(c, 0.0, 1.0))
    return StepGraphon(
        [c, 1.0 - c],
        [[float(a), float(d)], [float(d), float(b)]],
        {"family": "bipodal", **metadata},
    )


def clique_like(edge: float, **metadata: Any) -> StepGraphon:
    """Clique graphon with edge density ``edge`` and triangle density ``edge**1.5``."""

    edge = float(np.clip(edge, 0.0, 1.0))
    q = float(np.sqrt(edge))
    if q >= 1.0 - 1e-14:
        return erdos_renyi(1.0, family="clique-like", **metadata)
    if q <= 1e-14:
        return erdos_renyi(0.0, family="clique-like", **metadata)
    return StepGraphon(
        [q, 1.0 - q],
        [[1.0, 0.0], [0.0, 0.0]],
        {"family": "clique-like", "target_edge": edge, **metadata},
    )


def anti_clique_like(edge: float, **metadata: Any) -> StepGraphon:
    """Complement of a clique-like graphon with edge density ``edge``."""

    edge = float(np.clip(edge, 0.0, 1.0))
    return clique_like(1.0 - edge).complement({"family": "anti-clique-like", **metadata})


def turan_like(k: int, sizes: Sequence[float] | None = None, **metadata: Any) -> StepGraphon:
    """Complete multipartite graphon with zero within-pode probability."""

    if sizes is None:
        sizes = np.full(k, 1.0 / k)
    matrix = np.ones((k, k), dtype=float)
    np.fill_diagonal(matrix, 0.0)
    return StepGraphon(sizes, matrix, {"family": "turan-like", **metadata})


class GraphonFamily(ABC):
    """Base class for samplable graphon families."""

    name: str
    k: int
    optimizable: bool = False

    @abstractmethod
    def sample(
        self,
        n: int,
        seed: int | np.random.Generator | None = None,
        target: dict[str, float] | None = None,
    ) -> list[StepGraphon]:
        """Sample graphons from this family."""

    def initial_graphons(
        self,
        target: dict[str, float] | None = None,
        seed: int | np.random.Generator | None = None,
    ) -> list[StepGraphon]:
        graphons: list[StepGraphon] = []
        if target and "edge" in target:
            graphons.append(erdos_renyi(target["edge"], source="initial"))
        graphons.extend(self.sample(2, seed=seed, target=target))
        return graphons

    def random_vector(
        self,
        rng: np.random.Generator,
        target: dict[str, float] | None = None,
    ) -> NDArray[np.float64]:
        raise NotImplementedError(f"{self.name} is not optimizer-parameterized")

    def initial_vectors(
        self,
        starts: int,
        seed: int | np.random.Generator | None = None,
        target: dict[str, float] | None = None,
    ) -> list[NDArray[np.float64]]:
        rng = _rng(seed)
        return [self.random_vector(rng, target=target) for _ in range(starts)]

    def bounds(self) -> list[tuple[float, float]]:
        raise NotImplementedError(f"{self.name} is not optimizer-parameterized")

    def unpack_params(self, params: Sequence[float]) -> StepGraphon:
        raise NotImplementedError(f"{self.name} is not optimizer-parameterized")

    def pack_params(self, graphon: StepGraphon) -> NDArray[np.float64]:
        raise NotImplementedError(f"{self.name} is not optimizer-parameterized")


@dataclass
class ErdosRenyiFamily(GraphonFamily):
    name: str = "erdos-renyi"
    k: int = 1
    optimizable: bool = True

    def sample(
        self,
        n: int,
        seed: int | np.random.Generator | None = None,
        target: dict[str, float] | None = None,
    ) -> list[StepGraphon]:
        rng = _rng(seed)
        if target and "edge" in target:
            ps = np.full(n, target["edge"])
        else:
            ps = rng.uniform(0.0, 1.0, size=n)
        return [erdos_renyi(float(p)) for p in ps]

    def random_vector(
        self,
        rng: np.random.Generator,
        target: dict[str, float] | None = None,
    ) -> NDArray[np.float64]:
        p = float(target["edge"]) if target and "edge" in target else float(rng.uniform())
        return np.array([np.clip(p, 0.0, 1.0)], dtype=float)

    def initial_vectors(
        self,
        starts: int,
        seed: int | np.random.Generator | None = None,
        target: dict[str, float] | None = None,
    ) -> list[NDArray[np.float64]]:
        vectors = super().initial_vectors(starts, seed=seed, target=target)
        if target and "edge" in target:
            vectors.append(np.array([np.clip(target["edge"], 0.0, 1.0)], dtype=float))
        return vectors

    def bounds(self) -> list[tuple[float, float]]:
        return [(0.0, 1.0)]

    def unpack_params(self, params: Sequence[float]) -> StepGraphon:
        return erdos_renyi(float(params[0]))

    def pack_params(self, graphon: StepGraphon) -> NDArray[np.float64]:
        return np.array([graphon.edge_density()], dtype=float)


@dataclass
class SymmetricBipodalFamily(GraphonFamily):
    name: str = "symmetric-bipodal"
    k: int = 2
    optimizable: bool = True

    def sample(
        self,
        n: int,
        seed: int | np.random.Generator | None = None,
        target: dict[str, float] | None = None,
    ) -> list[StepGraphon]:
        rng = _rng(seed)
        values = rng.uniform(0.0, 1.0, size=(n, 2))
        return [symmetric_bipodal(a, d) for a, d in values]

    def random_vector(
        self,
        rng: np.random.Generator,
        target: dict[str, float] | None = None,
    ) -> NDArray[np.float64]:
        if target and "edge" in target:
            e = np.clip(float(target["edge"]), 0.0, 1.0)
            d = float(rng.uniform(0.0, 1.0))
            a = float(np.clip(2.0 * e - d, 0.0, 1.0))
            return np.array([a, d], dtype=float)
        return rng.uniform(0.0, 1.0, size=2)

    def initial_vectors(
        self,
        starts: int,
        seed: int | np.random.Generator | None = None,
        target: dict[str, float] | None = None,
    ) -> list[NDArray[np.float64]]:
        vectors = super().initial_vectors(starts, seed=seed, target=target)
        if target and "edge" in target:
            edge = float(np.clip(target["edge"], 0.0, 1.0))
            vectors.append(np.array([edge, edge], dtype=float))
            vectors.append(np.array([0.0, min(1.0, 2.0 * edge)], dtype=float))
            vectors.append(np.array([1.0, max(0.0, 2.0 * edge - 1.0)], dtype=float))
        return vectors

    def bounds(self) -> list[tuple[float, float]]:
        return [(0.0, 1.0), (0.0, 1.0)]

    def unpack_params(self, params: Sequence[float]) -> StepGraphon:
        a, d = np.clip(np.asarray(params, dtype=float), 0.0, 1.0)
        return symmetric_bipodal(float(a), float(d))

    def pack_params(self, graphon: StepGraphon) -> NDArray[np.float64]:
        if graphon.k != 2:
            raise ValueError("symmetric bipodal pack expects k=2")
        a = 0.5 * (graphon.matrix[0, 0] + graphon.matrix[1, 1])
        d = graphon.matrix[0, 1]
        return np.array([a, d], dtype=float)


@dataclass
class BipodalFamily(GraphonFamily):
    name: str = "bipodal"
    k: int = 2
    optimizable: bool = True
    min_size: float = 1e-8

    def sample(
        self,
        n: int,
        seed: int | np.random.Generator | None = None,
        target: dict[str, float] | None = None,
    ) -> list[StepGraphon]:
        rng = _rng(seed)
        graphons = []
        for _ in range(n):
            c = rng.uniform(self.min_size, 1.0 - self.min_size)
            a, b, d = rng.uniform(0.0, 1.0, size=3)
            graphons.append(bipodal(float(c), float(a), float(b), float(d)))
        return graphons

    def random_vector(
        self,
        rng: np.random.Generator,
        target: dict[str, float] | None = None,
    ) -> NDArray[np.float64]:
        c = float(rng.uniform(self.min_size, 1.0 - self.min_size))
        if target and "edge" in target and rng.uniform() < 0.35:
            e = float(np.clip(target["edge"], 0.0, 1.0))
            return np.array([c, e, e, e], dtype=float)
        return np.array([c, *rng.uniform(0.0, 1.0, size=3)], dtype=float)

    def initial_vectors(
        self,
        starts: int,
        seed: int | np.random.Generator | None = None,
        target: dict[str, float] | None = None,
    ) -> list[NDArray[np.float64]]:
        vectors = super().initial_vectors(starts, seed=seed, target=target)
        if target and "edge" in target:
            edge = float(np.clip(target["edge"], 0.0, 1.0))
            vectors.append(np.array([0.5, edge, edge, edge], dtype=float))
            q = float(np.sqrt(edge))
            if 0.0 < q < 1.0:
                vectors.append(np.array([q, 1.0, 0.0, 0.0], dtype=float))
        return vectors

    def bounds(self) -> list[tuple[float, float]]:
        return [(self.min_size, 1.0 - self.min_size), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

    def unpack_params(self, params: Sequence[float]) -> StepGraphon:
        c, a, b, d = np.asarray(params, dtype=float)
        return bipodal(
            float(np.clip(c, self.min_size, 1.0 - self.min_size)),
            float(np.clip(a, 0.0, 1.0)),
            float(np.clip(b, 0.0, 1.0)),
            float(np.clip(d, 0.0, 1.0)),
        )

    def pack_params(self, graphon: StepGraphon) -> NDArray[np.float64]:
        if graphon.k != 2:
            raise ValueError("bipodal pack expects k=2")
        return np.array(
            [graphon.sizes[0], graphon.matrix[0, 0], graphon.matrix[1, 1], graphon.matrix[0, 1]],
            dtype=float,
        )


@dataclass
class KPodalFamily(GraphonFamily):
    k: int
    name: str = "k-podal"
    optimizable: bool = True
    min_size: float = 1e-8
    matrix_sampler: str = "uniform"

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be positive")
        self.name = f"{self.k}-podal"

    @property
    def n_params(self) -> int:
        return self.k + upper_triangle_count(self.k)

    def _matrix_values(self, rng: np.random.Generator) -> NDArray[np.float64]:
        n = upper_triangle_count(self.k)
        if self.matrix_sampler == "boundary":
            return rng.beta(0.35, 0.35, size=n)
        if self.matrix_sampler == "beta":
            return rng.beta(2.0, 2.0, size=n)
        return rng.uniform(0.0, 1.0, size=n)

    def sample(
        self,
        n: int,
        seed: int | np.random.Generator | None = None,
        target: dict[str, float] | None = None,
    ) -> list[StepGraphon]:
        rng = _rng(seed)
        graphons = []
        for _ in range(n):
            sizes = rng.dirichlet(np.ones(self.k))
            matrix = symmetric_from_upper(self.k, self._matrix_values(rng))
            graphons.append(StepGraphon(sizes, matrix, {"family": self.name}))
        return graphons

    def random_vector(
        self,
        rng: np.random.Generator,
        target: dict[str, float] | None = None,
    ) -> NDArray[np.float64]:
        sizes = rng.dirichlet(np.ones(self.k))
        if target and "edge" in target and rng.uniform() < 0.25:
            p = np.full(upper_triangle_count(self.k), np.clip(target["edge"], 0.0, 1.0))
        else:
            p = self._matrix_values(rng)
        return np.concatenate([sizes, p]).astype(float)

    def initial_vectors(
        self,
        starts: int,
        seed: int | np.random.Generator | None = None,
        target: dict[str, float] | None = None,
    ) -> list[NDArray[np.float64]]:
        rng = _rng(seed)
        vectors = [self.random_vector(rng, target=target) for _ in range(starts)]
        if target and "edge" in target:
            edge = float(np.clip(target["edge"], 0.0, 1.0))
            structured = [
                erdos_renyi(edge, source="initial"),
                clique_like(edge, source="initial"),
                anti_clique_like(edge, source="initial"),
            ]
            if self.k >= 2:
                structured.append(turan_like(self.k, source="initial"))
            for graphon in structured:
                vectors.append(self.pack_params(self._pad_graphon(graphon)))
        return vectors

    def _pad_graphon(self, graphon: StepGraphon) -> StepGraphon:
        if graphon.k == self.k:
            return graphon
        if graphon.k > self.k:
            raise ValueError("cannot pack graphon with larger k")
        extra = self.k - graphon.k
        pad_mass = min(0.05, extra * self.min_size * 10.0)
        pad_mass = max(pad_mass, extra * self.min_size)
        sizes = np.full(self.k, pad_mass / extra if extra else 0.0)
        sizes[: graphon.k] = graphon.sizes * (1.0 - pad_mass)
        matrix = np.full((self.k, self.k), graphon.edge_density(), dtype=float)
        matrix[: graphon.k, : graphon.k] = graphon.matrix
        return StepGraphon(sizes, matrix, {"family": self.name, "padded_from": graphon.metadata.get("family")})

    def bounds(self) -> list[tuple[float, float]]:
        return [(self.min_size, 1.0)] * self.k + [(0.0, 1.0)] * upper_triangle_count(self.k)

    def unpack_params(self, params: Sequence[float]) -> StepGraphon:
        params = np.asarray(params, dtype=float)
        if params.size != self.n_params:
            raise ValueError(f"expected {self.n_params} parameters, got {params.size}")
        raw_sizes = np.clip(params[: self.k], self.min_size, 1.0)
        sizes = raw_sizes / np.sum(raw_sizes)
        matrix = symmetric_from_upper(self.k, np.clip(params[self.k :], 0.0, 1.0))
        return StepGraphon(sizes, matrix, {"family": self.name})

    def pack_params(self, graphon: StepGraphon) -> NDArray[np.float64]:
        if graphon.k != self.k:
            graphon = self._pad_graphon(graphon)
        return np.concatenate([graphon.sizes, upper_triangle_values(graphon.matrix)]).astype(float)


@dataclass
class CliqueLikeFamily(GraphonFamily):
    name: str = "clique-like"
    k: int = 2
    optimizable: bool = False

    def sample(
        self,
        n: int,
        seed: int | np.random.Generator | None = None,
        target: dict[str, float] | None = None,
    ) -> list[StepGraphon]:
        rng = _rng(seed)
        if target and "edge" in target:
            edges = np.full(n, target["edge"])
        else:
            edges = rng.uniform(0.0, 1.0, size=n)
        return [clique_like(float(e)) for e in edges]


@dataclass
class AntiCliqueLikeFamily(GraphonFamily):
    name: str = "anti-clique-like"
    k: int = 2
    optimizable: bool = False

    def sample(
        self,
        n: int,
        seed: int | np.random.Generator | None = None,
        target: dict[str, float] | None = None,
    ) -> list[StepGraphon]:
        rng = _rng(seed)
        if target and "edge" in target:
            edges = np.full(n, target["edge"])
        else:
            edges = rng.uniform(0.0, 1.0, size=n)
        return [anti_clique_like(float(e)) for e in edges]


@dataclass
class TuranLikeFamily(GraphonFamily):
    k: int
    name: str = "turan-like"
    optimizable: bool = False

    def sample(
        self,
        n: int,
        seed: int | np.random.Generator | None = None,
        target: dict[str, float] | None = None,
    ) -> list[StepGraphon]:
        rng = _rng(seed)
        return [turan_like(self.k, rng.dirichlet(np.ones(self.k))) for _ in range(n)]


def default_sampling_families(kmax: int) -> list[GraphonFamily]:
    families: list[GraphonFamily] = [
        ErdosRenyiFamily(),
        SymmetricBipodalFamily(),
        BipodalFamily(),
        CliqueLikeFamily(),
        AntiCliqueLikeFamily(),
    ]
    families.extend(KPodalFamily(k) for k in range(1, kmax + 1))
    families.extend(TuranLikeFamily(k) for k in range(2, kmax + 1))
    return families


def default_optimization_families(kmax: int, include_er: bool = False) -> list[GraphonFamily]:
    families: list[GraphonFamily] = []
    if include_er:
        families.append(ErdosRenyiFamily())
    families.extend([SymmetricBipodalFamily(), BipodalFamily()])
    families.extend(KPodalFamily(k) for k in range(3, kmax + 1))
    return families


def family_by_name(name: str, k: int | None = None) -> GraphonFamily:
    key = name.strip().lower()
    if key in {"er", "erdos-renyi", "erdos_renyi"}:
        return ErdosRenyiFamily()
    if key in {"symmetric-bipodal", "sym-bipodal", "symmetric_bipodal"}:
        return SymmetricBipodalFamily()
    if key in {"bipodal", "general-bipodal"}:
        return BipodalFamily()
    if key in {"kpodal", "k-podal", "podal"}:
        if k is None:
            raise ValueError("k is required for k-podal family")
        return KPodalFamily(k)
    if key in {"clique", "clique-like"}:
        return CliqueLikeFamily()
    if key in {"anti-clique", "anti-clique-like"}:
        return AntiCliqueLikeFamily()
    if key in {"turan", "turan-like"}:
        if k is None:
            raise ValueError("k is required for Turan-like family")
        return TuranLikeFamily(k)
    raise ValueError(f"unknown family {name!r}")


def all_family_names() -> list[str]:
    return [
        "erdos-renyi",
        "symmetric-bipodal",
        "bipodal",
        "k-podal",
        "clique-like",
        "anti-clique-like",
        "turan-like",
    ]


def sample_from_families(
    families: Iterable[GraphonFamily],
    samples_per_family: int,
    seed: int | None = None,
    target: dict[str, float] | None = None,
) -> list[StepGraphon]:
    rng = np.random.default_rng(seed)
    graphons: list[StepGraphon] = []
    for family in families:
        child_seed = int(rng.integers(0, 2**32 - 1))
        graphons.extend(family.sample(samples_per_family, seed=child_seed, target=target))
    return graphons
