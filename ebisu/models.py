from dataclasses import dataclass, field
from typing import Union
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


@dataclass
class Predict(DataClassJsonMixin):
  version: int
  lastEncounterMs: float  # milliseconds since unix epoch
  wmax: float  # weight for max halflife (it's actually the smallest weight), between 0 and 1
  hmax: float  # max halflife, in hours
  log2ws: list[float]
  hs: list[float]  # same length as log2ws
  # recall probability is proportional to:
  # `MAX(log2ws - ((NOW_MS - lastEncounterMs) * HOURS_PER_MILLISECONDS / hs)`
  # where NOW_MS is milliseconds since Unix epoch.


@dataclass
class Model(DataClassJsonMixin):
  version: int
  quiz: Quiz
  pred: Predict


def success(res: Result) -> bool:
  if isinstance(res, NoisyBinaryResult):
    return res.result > 0.5
  elif isinstance(res, BinomialResult):
    return res.successes * 2 > res.total
  else:
    raise Exception("unknown result type")
