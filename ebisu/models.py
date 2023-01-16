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

  # 0 < x <= 1 (reinforcement). Same length/sub-lengths as `results`
  startStrengths: list[list[float]]

  # same length as `results`. Timestamp of the first item in each sub-array of
  # `results`
  startTimestampMs: list[float]


@dataclass
class Probability(DataClassJsonMixin):
  version: int
  # priors: fixed at model creation time
  initHlPrior: tuple[float, float]  # alpha and beta
  boostPrior: tuple[float, float]  # alpha and beta

  # posteriors: these change after quizzes
  initHl: tuple[float, float]  # alpha and beta
  boost: tuple[float, float]  # alpha and beta
  # we need both prior (belief before any quizzes are received) and posterior
  # (after quizzes) because when updating using all history
  # (`updateRecallHistory`), we need to know what to start from.


@dataclass
class Predict(DataClassJsonMixin):
  version: int
  # just for developer ease, these can be stored in SQL, etc.
  lastEncounterMs: float  # milliseconds since unix epoch
  currentHalflifeHours: float  # mean (so _currentHalflifePrior works). Same units as `elapseds`
  logStrength: float
  # recall probability is proportional to:
  # `logStrength - ((NOW_MS - lastEncounterMs) * HOURS_PER_MILLISECONDS) / currentHalflifeHours`
  # where NOW_MS is milliseconds since Unix epoch.


@dataclass
class Model(DataClassJsonMixin):
  quiz: Quiz
  prob: Probability
  pred: Predict


@dataclass
class GammaUpdate:
  a: float
  b: float
  mean: float