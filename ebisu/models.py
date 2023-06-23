from dataclasses import dataclass, field
from typing import Optional, Union
from dataclasses_json import DataClassJsonMixin, config


@dataclass
class BinomialResult(DataClassJsonMixin):
  successes: int
  total: int
  hoursElapsed: float
  rescale: Union[float, int]


@dataclass
class NoisyBinaryResult(DataClassJsonMixin):
  result: float
  q1: float
  q0: float
  hoursElapsed: float
  rescale: Union[float, int]


Result = Union[BinomialResult, NoisyBinaryResult]


def cleanup(results):
  """Workaround for dataclasses-json unable to handle Union inside List
  
  See https://github.com/lidatong/dataclasses-json/issues/239"""
  return [NoisyBinaryResult(**res) if 'q1' in res else BinomialResult(**res) for res in results]


HalflifeGamma = tuple[float, float]  # α, β


@dataclass
class Model(DataClassJsonMixin):
  version: int
  results: list[Result] = field(metadata=config(decoder=cleanup))
  startTimestampMs: float  # milliseconds since unix epoch
  lastEncounterMs: float  # milliseconds since unix epoch
  power: int
  log2weights: list[float]
  halflifeGammas: list[HalflifeGamma]  # same length as log2weights

  forSql: list[tuple[float, float]]  # `(log2weight, halflifeInMillisecond)`
  # recall probability is proportional to:
  # `MAX(log2weights - ((NOW_MS - lastEncounterMs) * HOURS_PER_MILLISECONDS / halflives)`
  # where NOW_MS is milliseconds since Unix epoch.
