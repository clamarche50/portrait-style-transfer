"""In-memory debug artifact collection with deterministic names."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass
class ArtifactCollector:
    enabled: bool = False
    artifacts: dict[str, NDArray[np.float32]] = field(default_factory=dict)

    def add(self, name: str, value: ArrayLike) -> None:
        if not self.enabled:
            return
        if not name or name in self.artifacts:
            raise ValueError("artifact names must be non-empty and unique")
        self.artifacts[name] = np.asarray(value, dtype=np.float32).copy()

    def snapshot(self) -> dict[str, NDArray[np.float32]]:
        return {name: value.copy() for name, value in self.artifacts.items()}
