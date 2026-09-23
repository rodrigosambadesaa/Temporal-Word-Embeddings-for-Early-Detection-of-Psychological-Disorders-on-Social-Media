from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Literal

import numpy as np
import os
import pandas as pd
import time
import torch
from scipy.sparse import dok_matrix
from tqdm import tqdm

from framework3.base.base_clases import BaseFilter
from framework3.base.base_types import XYData
from framework3.container import Container

from src.models.deltas import DISTANCES
from src.models.twec import TWEC


DistanceName = Literal[
    "cosine",
    "euclidean",
    "chebyshev",
    "jensen_shannon",
    "wasserstein",
    "manhattan",
    "minkowski",
]


@Container.bind()
class TWECFilter(BaseFilter):
    def __init__(
        self,
        context_size: int,
        _cpus: int = 4,
        deltas_f: List[DistanceName] | None = None,
    ):
        super().__init__()
        if context_size <= 0:
            raise ValueError("context_size must be positive")
        if _cpus <= 0:
            raise ValueError("_cpus must be positive")

        # Slice-level parallelism is handled by ThreadPoolExecutor below. Using one
        # Word2Vec worker per slice avoids severe nested oversubscription.
        self._twec = TWEC(size=300, window=context_size, workers=1)
        self.deltas_f = list(deltas_f or ["cosine"])
        unknown = set(self.deltas_f) - set(DISTANCES)
        if unknown:
            raise ValueError(f"Unsupported distance metrics: {sorted(unknown)}")

        self.context_size = context_size
        self._cpus = min(os.cpu_count() or 1, _cpus)
        self._vocab_hash_map: dict[str, int] = {}

    def fit(self, x: XYData, y: XYData | None) -> float | None:
        del y
        start = time.time()
        data: pd.DataFrame = x.value
        if data.empty:
            raise ValueError("TWECFilter cannot be fitted on an empty dataset")

        self._twec.train_compass(data.text.values.tolist())
        if self._twec.compass is None:
            raise RuntimeError("TWEC compass training did not produce a model")

        self._vocab_hash_map = {
            word: index
            for index, word in enumerate(self._twec.compass.wv.index_to_key)
        }
        print(f"* TWEC training took: {time.time() - start:.6f} seconds")
        return None

    def predict(self, x: XYData) -> XYData:
        if self._twec.compass is None or not self._vocab_hash_map:
            raise RuntimeError("TWECFilter must be fitted before predict()")

        start = time.time()
        data: pd.DataFrame = x.value
        n_rows = len(data.index)
        n_cols = len(self._vocab_hash_map)
        metric_names = self.deltas_f

        all_deltas = {
            metric: dok_matrix((n_rows, n_cols), dtype=np.float32)
            for metric in metric_names
        }

        def process_user_deltas(i, texts):
            temporal_model = self._twec.train_slice(texts)
            result = {metric: [] for metric in metric_names}

            for word in temporal_model.wv.index_to_key:
                j = self._vocab_hash_map.get(word)
                if j is None:
                    continue

                compass_vector = torch.as_tensor(
                    self._twec.compass.wv[word], dtype=torch.float32
                ).reshape(1, 1, -1)
                temporal_vector = torch.as_tensor(
                    temporal_model.wv[word], dtype=torch.float32
                ).reshape(1, 1, -1)

                for metric in metric_names:
                    distance = DISTANCES[metric](
                        compass_vector, temporal_vector
                    ).detach().cpu().item()
                    result[metric].append((i, j, float(distance)))
            return result

        with ThreadPoolExecutor(max_workers=self._cpus) as executor:
            futures = {
                executor.submit(process_user_deltas, i, row.text): i
                for i, row in tqdm(
                    enumerate(data.itertuples()),
                    total=n_rows,
                    desc="Scheduling temporal slices",
                )
            }

            for future in tqdm(
                as_completed(futures), total=n_rows, desc="Computing temporal deltas"
            ):
                chunk_result = future.result()
                for metric, values in chunk_result.items():
                    for i, j, value in values:
                        all_deltas[metric][i, j] = value

        print(f"* TWEC prediction took: {time.time() - start:.6f} seconds")
        return XYData.mock(all_deltas)
