import pandas as pd


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer

        self._data = pd.DataFrame(
            index=keys,
            columns=["total", "counts", "average"],
            dtype="float64",
        )
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col] = 0.0

    def update(self, key, value, n=1):
        value = float(value)

        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = (
            self._data.total[key] / self._data.counts[key]
        )

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def keys(self):
        return self._data.total.keys()