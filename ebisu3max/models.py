from dataclasses import dataclass, field
from typing import Optional, Union
from dataclasses_json import DataClassJsonMixin, config


@dataclass
class BinomialResult(DataClassJsonMixin):
  successes: int
  total: int
  hoursElapsed: float


@dataclass
class NoisyBinaryResult(DataClassJsonMixin):
  result: float
  q1: float
  q0: float
  hoursElapsed: float


Result = Union[BinomialResult, NoisyBinaryResult]


def cleanup(results):
  """Workaround for dataclasses-json unable to handle Union inside List
  
  See https://github.com/lidatong/dataclasses-json/issues/239"""
  return [[NoisyBinaryResult(**res) if 'q1' in res else BinomialResult(**res)
           for res in lst]
          for lst in results]


@dataclass
class Quiz(DataClassJsonMixin):
  version: int
  results: list[list[Result]] = field(metadata=config(decoder=cleanup))

  # same length as `results`. Timestamp of the first item in each sub-array of
  # `results`
  startTimestampMs: list[float]


HalflifeGamma = tuple[float, float]  # α, β


@dataclass
class Predict(DataClassJsonMixin):
  version: int
  lastEncounterMs: float  # milliseconds since unix epoch
  log2weights: list[float]
  halflifeGammas: list[HalflifeGamma]  # same length as log2weights
  weightsReached: list[bool]  # same length as log2weights

  forSql: list[tuple[float, float]]  # `(log2weight, halflifeInMillisecond)`
  # recall probability is proportional to:
  # `MAX(log2weights - ((NOW_MS - lastEncounterMs) * HOURS_PER_MILLISECONDS / halflives)`
  # where NOW_MS is milliseconds since Unix epoch.
  power: Optional[int]


@dataclass
class Model(DataClassJsonMixin):
  version: int
  quiz: Quiz
  pred: Predict
