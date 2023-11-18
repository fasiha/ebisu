from math import fsum, isfinite, log, log10, log2, isfinite
import numpy as np
from scipy.stats import binom  # type:ignore
from scipy.optimize import minimize_scalar  # type:ignore

from typing import Optional, Union, List
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from . import ebisu2beta as ebisu2
from ebisu.logsumexp import logsumexp, sumexp

LN2 = log(2)


@dataclass_json
@dataclass
class Atom:
  "An atom's weight, Beta params, and time horizon"
  log2weight: float
  alpha: float
  beta: float
  time: float


BetaEnsemble = List[Atom]


def initModel(
    firstHalflife: float,
    lastHalflife: Optional[float] = None,  # default will be `10_000 * firstHalflife`
    firstWeight: float = 0.9,
    numAtoms: int = 5,
    initialAlphaBeta: float = 2.0,
) -> BetaEnsemble:
  """Initialize an Ebisu model

  Run this when the student first learns a new flashcard.
  `firstHalflife` is your best guess for how long before this
  flashcard's memory drops to 50%, in units of your choice. `numAtoms`
  atoms will be created from this halflife to `lastHalflife`, each
  initialized with the same Beta random variable, parameterized as
  $Beta(α, β)$ with `α=β=initialAlphaBeta`, which should be >1 and
  probably >= 2.

  Having a lot of atoms takes more CPU and can harm prediction accuracy,
  so there's no reason to make the ensemble between the first and last
  halflife very dense.

  `lastHalflife` probably should be something on the order of years
  since there are no atoms beyond it to prevent the recall probability
  from falling off steeply beyond it.

  Each atom is weighted, with the first atom (corresponding to
  `firstHalflife`) getting `firstWeight` and the rest of the atoms
  getting logarithmically-less weight.

  The recall probability of the ensemble is the weighted sum of the
  individual atoms' recall probability. Similarly, the ensemble's update
  is the result of updating each atom (modulo some rules to help avoid
  destroying low-weight long-halflife atoms by early failures).
  """
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
  # Now `sum(weights)` ≈ 1, and they are logarithmically spaced, i.e.,
  # `np.logspace(np.log10(weights[0]), np.log10(weights[-1]), numAtoms)` ≈ `weights`.

  wsum = fsum(weights)
  log2weights = [log2(w / wsum) for w in weights]

  halflives = np.logspace(
      log10(firstHalflife), log10(lastHalflife or firstHalflife * 1e4), numAtoms)

  return [
      Atom(log2weight=l2w, alpha=initialAlphaBeta, beta=initialAlphaBeta, time=h)
      for l2w, h in zip(log2weights, halflives)
  ]


def predictRecall(model: BetaEnsemble, elapsedTime: float) -> float:
  """The current probability of recall

  Given an Ebisu model (created by `initModel`), and how much time (in
  consistent units with `initModel`) has elapsed, return the probability
  of recall.
  """
  logps = [
      LN2 * m.log2weight + ebisu2.predictRecall(
          (m.alpha, m.beta, m.time), tnow=elapsedTime, exact=False) for m in model
  ]
  result = sumexp(logps)
  assert isfinite(result) and 0 <= result <= 1
  return result


def predictRecallApprox(prior: BetaEnsemble, elapsedTime: float) -> float:
  "A fast approximation of `predictRecall`"
  logps = [LN2 * (m.log2weight - elapsedTime / m.time) for m in prior]
  return sumexp(logps)


def modelToPercentileDecay(model: BetaEnsemble, percentile=0.5) -> float:
  "When will this model's recall probability to decay to `percentile`?"
  assert (0 < percentile <= 1), "percentile must be in (0, 1]"

  logLeft, logRight = 1, 2
  counter = 0
  while predictRecall(model, 10**logLeft) <= 0.5:
    logLeft -= 1
    counter += 1
    if counter >= 10:
      raise Exception('unable to find left bound')

  counter = 0
  while predictRecall(model, 10**logRight) >= 0.5:
    logRight += 1
    counter += 1
    if counter >= 10:
      raise Exception('unable to find right bound')

  res = minimize_scalar(
      lambda h: abs(percentile - predictRecall(model, h)), bracket=[10**logLeft, 10**logRight])
  assert res.success
  return res.x


def _noisyLogProbability(result: Union[int, float], q1: float, q0: float, p: float) -> float:
  z = result >= 0.5
  return log(((q1 - q0) * p + q0) if z else (q0 - q1) * p + (1 - q0))


def _binomialLogProbability(successes: int, total: int, p: float) -> float:
  return float(binom.logpmf(successes, total, p))


def updateRecall(
    model: BetaEnsemble,
    successes: Union[float, int],
    total: int,
    elapsedTime: float,
    q0: Optional[float] = None,
    updateThreshold=0.9,
    weightThreshold=0.9,
) -> BetaEnsemble:
  """Update an Ebisu model with a quiz's resuilts

  Update an existing `model` (e.g., returned by `initModel` or
  `updateRecall` itself) with a binomial quiz for integers `0 <=
  successes <= total > 1`, or noisy-binary quiz when `total=1` and `0 <
  successes < 1` is a float, assuming the quiz happened `elapsedTime`
  time units after its last encounter. (See Ebisu math docs on what
  "binomial" and "noisy-binary" mean.)

  `q0` (defaults to `1-q1` where `q1=max(successes, 1-successes)`) works
  with `successes` for the noisy-quiz case.

  `updateThreshold` and `weightThreshold` govern whether
  low-enough-weight atoms will be impacted by failed quizzes, in order
  to avoid very-long-halflife atoms from being disproportionately
  impacted by early failures. Though all atoms' weights will be
  reweighted by the quiz results, only atoms whose `newHalflife /
  oldHalflife > updateThreshold` OR atoms whose weights
  (cumulatively-summed from the last atom to the first) that exceed
  `weightThreshold` will be updated with the quiz result.
  """
  assert total == int(total) and total >= 1, "total must be a positive integer"
  updatedModels = [
      ebisu2.updateRecall((m.alpha, m.beta, m.time), successes, total, tnow=elapsedTime, q0=q0)
      for m in model
  ]
  # That's the updated Beta models! Now we need to update weights.

  pRecalls = [
      ebisu2.predictRecall((m.alpha, m.beta, m.time), elapsedTime, exact=True) for m in model
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

  assert all(x <= 0 for x in individualLogProbabilities
            ), f'{individualLogProbabilities=}, {model=}, {elapsedTime=}'

  newAtoms: BetaEnsemble = []
  for (
      oldAtom,
      updatedAtom,
      lp,
      exceededWeight,
  ) in zip(
      model,
      updatedModels,
      individualLogProbabilities,
      _exceedsThresholdLeft([2**x.log2weight for x in model], weightThreshold),
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


def _exceedsThresholdLeft(v: list[float], threshold: float) -> list[bool]:
  return (np.cumsum(v[::-1])[::-1] > threshold).tolist()


def rescaleHalflife(model: BetaEnsemble, scale: float) -> BetaEnsemble:
  "Rescale every atom's halflife by `scale`"
  assert scale > 0, "positive scale required"
  ret: BetaEnsemble = []
  for atom in model:
    scaled = ebisu2.rescaleHalflife((atom.alpha, atom.beta, atom.time), scale)
    ret.append(Atom(log2weight=atom.log2weight, alpha=scaled[0], beta=scaled[1], time=scaled[2]))
  return ret
