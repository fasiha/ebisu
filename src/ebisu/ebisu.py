from math import fsum, isfinite, log, log10, log2, isfinite
import numpy as np
from scipy.stats import binom  # type:ignore
from scipy.optimize import minimize_scalar  # type:ignore

from typing import Optional, Union
from attr import dataclass
from dataclasses_json import DataClassJsonMixin

from . import ebisu2beta as ebisu2
from ebisu.logsumexp import logsumexp

LN2 = log(2)


@dataclass
class Atom(DataClassJsonMixin):
  log2weight: float
  alpha: float
  beta: float
  time: float


BetaEnsemble = list[Atom]


def initModel(
    firstHalflife: float,
    lastHalflife: float,
    firstWeight: float = 0.9,
    numAtoms: int = 5,
    initialAlphaBeta: float = 2.0,
) -> BetaEnsemble:
  # We have `numAtoms` weights which have to sum to 1. The first one is `firstWeight`. We want the
  # rest of them to be logarithmically-decaying. So we need to find $d$ such that
  #
  # $1 = ∑_{i=1}^n (w_1)^(d * (i-1))$
  #
  # where $w_1$ is `firstWeight`. As far as I know, we can't solve this analytically so we do a
  # quick search.
  solution = minimize_scalar(lambda d: abs(sum(firstWeight * d**i for i in range(numAtoms)) - 1),
                             [0.3, 0.6])
  weights = [firstWeight * solution.x**i for i in range(numAtoms)]
  # `sum(weights)` ≈ 1, and they are logarithmically spaced, i.e.,
  # `np.logspace(np.log10(weights[0]), np.log10(weights[-1]), numAtoms)` ≈ `weights`.

  halflives = np.logspace(log10(firstHalflife), log10(lastHalflife), numAtoms)

  wsum = fsum(weights)
  log2weights = [log2(w / wsum) for w in weights]

  return [
      Atom(log2weight=l2w, alpha=initialAlphaBeta, beta=initialAlphaBeta, time=h)
      for l2w, h in zip(log2weights, halflives)
  ]


def predictRecall(
    prior: BetaEnsemble,
    elapsedTime: float,
    logDomain=True,
) -> float:
  logps = [
      LN2 * m.log2weight + ebisu2.predictRecall(
          (m.alpha, m.beta, m.time), tnow=elapsedTime, exact=False) for m in prior
  ]
  log2Expect = logsumexp(logps) / LN2

  assert isfinite(log2Expect) and log2Expect <= 0, f'{logps=}, {log2Expect=}'
  return log2Expect if logDomain else 2**log2Expect


def predictRecallApprox(
    prior: BetaEnsemble,
    elapsedTime: float,
    logDomain=True,
) -> float:
  logps = [LN2 * (m.log2weight - elapsedTime / m.time) for m in prior]
  log2Expect = logsumexp(logps) / LN2

  assert isfinite(log2Expect) and log2Expect <= 0, f'{logps=}, {log2Expect=}'
  return log2Expect if logDomain else 2**log2Expect


def modelToPercentileDecay(model: BetaEnsemble, percentile=0.5) -> float:
  "When will this model's recall probability to decay to `percentile`?"
  assert (0 < percentile <= 1), "percentile must be in (0, 1]"
  l2p = log2(percentile)

  res = minimize_scalar(
      lambda h: abs(l2p - predictRecall(model, h, logDomain=True)), bounds=[.01, 100e3])
  assert res.success
  return res.x


def _noisyLogProbability(result: Union[int, float], q1: float, q0: float, p: float) -> float:
  z = result >= 0.5
  return log(((q1 - q0) * p + q0) if z else (q0 - q1) * p + (1 - q0))


def _binomialLogProbability(successes: int, total: int, p: float) -> float:
  return float(binom.logpmf(successes, total, p))


def updateRecall(
    prior: BetaEnsemble,
    successes: Union[float, int],
    total: int,
    elapsedTime: float,
    q0: Optional[float] = None,
    updateThreshold: Optional[float] = None,
    weightThreshold: Optional[float] = None,
) -> BetaEnsemble:
  updateThreshold = updateThreshold if updateThreshold is not None else 0.99
  weightThreshold = weightThreshold if weightThreshold is not None else 0.49
  updatedModels = [
      ebisu2.updateRecall((m.alpha, m.beta, m.time), successes, total, tnow=elapsedTime, q0=q0)
      for m in prior
  ]
  # That's the updated Beta models! Now we need to update weights.

  pRecalls = [
      ebisu2.predictRecall((m.alpha, m.beta, m.time), elapsedTime, exact=True) for m in prior
  ]
  if total == 1:
    # some of this is repeated from ebisu2beta
    q1 = max(successes, 1 - successes)  # between 0.5 and 1
    q0 = 1 - q1 if q0 is None else q0  # either the input argument OR between 0 and 0.5

    individualLogProbabilities = [
        _noisyLogProbability(successes, q1=q1, q0=q0, p=p) for p in pRecalls
    ]

  else:
    individualLogProbabilities = [
        _binomialLogProbability(int(successes), total=total, p=p) for p in pRecalls
    ]

  assert all(x < 0 for x in individualLogProbabilities
            ), f'{individualLogProbabilities=}, {prior=}, {elapsedTime=}'

  newAtoms: BetaEnsemble = []
  for (
      oldAtom,
      updatedAtom,
      lp,
      exceededWeight,
  ) in zip(
      prior,
      updatedModels,
      individualLogProbabilities,
      _exceedsThresholdLeft([x.log2weight for x in prior], log2(weightThreshold)),
  ):
    oldHl = oldAtom.time
    newHl = updatedAtom[-1]
    scal = newHl / oldHl

    newLog2Weight = (oldAtom.log2weight + lp / LN2)  # particle filter update, we'll normalize later
    if scal > updateThreshold or exceededWeight:
      newAtoms.append(
          Atom(
              alpha=updatedAtom[0],
              beta=updatedAtom[1],
              time=updatedAtom[2],
              log2weight=newLog2Weight))
    else:
      newAtoms.append(
          Atom(alpha=oldAtom.alpha, beta=oldAtom.beta, time=oldAtom.time, log2weight=newLog2Weight))

  # equivalent to
  # ```py
  # wsum = sum(2**m.log2weight for m in newAtoms)
  # for atom in newAtoms:
  #   atom.log2weight = log2(2**atom.log2weight / wsum)
  # ```
  log2WeightSum = logsumexp([m.log2weight * LN2 for m in newAtoms]) / LN2
  for atom in newAtoms:
    atom.log2weight -= log2WeightSum

  return newAtoms


def _exceedsThresholdLeft(v: list[float], threshold: float):
  ret: list[bool] = []
  last = False
  for x in v[::-1]:
    last = last or x > threshold
    ret.append(last)
  return ret[::-1]


def rescaleHalflife(model: BetaEnsemble, scale: float) -> BetaEnsemble:
  ret: BetaEnsemble = []
  for atom in model:
    scaled = ebisu2.rescaleHalflife((atom.alpha, atom.beta, atom.time), scale)
    ret.append(Atom(log2weight=atom.log2weight, alpha=scaled[0], beta=scaled[1], time=scaled[2]))
  return ret
