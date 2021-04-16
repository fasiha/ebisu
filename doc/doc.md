<a name="ebisu"></a>
# ebisu

<a name="ebisu.alternate"></a>
# ebisu.alternate

<a name="ebisu.alternate.predictRecallMode"></a>
#### predictRecallMode

```python
predictRecallMode(prior, tnow)
```

Mode of the immediate recall probability.

Same arguments as `ebisu.predictRecall`, see that docstring for details. A
returned value of 0 or 1 may indicate divergence.

<a name="ebisu.alternate.predictRecallMedian"></a>
#### predictRecallMedian

```python
predictRecallMedian(prior, tnow, percentile=0.5)
```

Median (or percentile) of the immediate recall probability.

Same arguments as `ebisu.predictRecall`, see that docstring for details.

An extra keyword argument, `percentile`, is a float between 0 and 1, and
specifies the percentile rather than 50% (median).

<a name="ebisu.alternate.predictRecallMonteCarlo"></a>
#### predictRecallMonteCarlo

```python
predictRecallMonteCarlo(prior, tnow, N=1000 * 1000)
```

Monte Carlo simulation of the immediate recall probability.

Same arguments as `ebisu.predictRecall`, see that docstring for details. An
extra keyword argument, `N`, specifies the number of samples to draw.

This function returns a dict containing the mean, variance, median, and mode
of the current recall probability.

<a name="ebisu.alternate.updateRecallMonteCarlo"></a>
#### updateRecallMonteCarlo

```python
updateRecallMonteCarlo(prior, k, n, tnow, tback=None, N=10 * 1000 * 1000, q0=None)
```

Update recall probability with quiz result via Monte Carlo simulation.

Same arguments as `ebisu.updateRecall`, see that docstring for details.

An extra keyword argument `N` specifies the number of samples to draw.

<a name="ebisu.ebisu"></a>
# ebisu.ebisu

<a name="ebisu.ebisu.predictRecall"></a>
#### predictRecall

```python
predictRecall(prior, tnow, exact=False)
```

Expected recall probability now, given a prior distribution on it. 🍏

`prior` is a tuple representing the prior distribution on recall probability
after a specific unit of time has elapsed since this fact's last review.
Specifically,  it's a 3-tuple, `(alpha, beta, t)` where `alpha` and `beta`
parameterize a Beta distribution that is the prior on recall probability at
time `t`.

`tnow` is the *actual* time elapsed since this fact's most recent review.

Optional keyword parameter `exact` makes the return value a probability,
specifically, the expected recall probability `tnow` after the last review: a
number between 0 and 1. If `exact` is false (the default), some calculations
are skipped and the return value won't be a probability, but can still be
compared against other values returned by this function. That is, if

> predictRecall(prior1, tnow1, exact=True) < predictRecall(prior2, tnow2, exact=True)

then it is guaranteed that

> predictRecall(prior1, tnow1, exact=False) < predictRecall(prior2, tnow2, exact=False)

The default is set to false for computational efficiency.

See README for derivation.

<a name="ebisu.ebisu.binomln"></a>
#### binomln

```python
binomln(n, k)
```

Log of scipy.special.binom calculated entirely in the log domain

<a name="ebisu.ebisu.updateRecall"></a>
#### updateRecall

```python
updateRecall(prior, successes, total, tnow, rebalance=True, tback=None, q0=None)
```

Update a prior on recall probability with a quiz result and time. 🍌

`prior` is same as in `ebisu.predictRecall`'s arguments: an object
representing a prior distribution on recall probability at some specific time
after a fact's most recent review.

`successes` is the number of times the user *successfully* exercised this
memory during this review session, out of `n` attempts. Therefore, `0 <=
successes <= total` and `1 <= total`.

If the user was shown this flashcard only once during this review session,
then `total=1`. If the quiz was a success, then `successes=1`, else
`successes=0`. (See below for fuzzy quizzes.)

If the user was shown this flashcard *multiple* times during the review
session (e.g., Duolingo-style), then `total` can be greater than 1.

If `total` is 1, `successes` can be a float between 0 and 1 inclusive. This
implies that while there was some "real" quiz result, we only observed a
scrambled version of it, which is `successes > 0.5`. A "real" successful quiz
has a `max(successes, 1 - successes)` chance of being scrambled such that we
observe a failed quiz `successes > 0.5`. E.g., `successes` of 0.9 *and* 0.1
imply there was a 10% chance a "real" successful quiz could result in a failed
quiz.

This noisy quiz model also allows you to specify the related probability that
a "real" quiz failure could be scrambled into the successful quiz you observed.
Consider "Oh no, if you'd asked me that yesterday, I would have forgotten it."
By default, this probability is `1 - max(successes, 1 - successes)` but doesn't
need to be that value. Provide `q0` to set this explicitly. See the full Ebisu
mathematical analysis for details on this model and why this is called "q0".

`tnow` is the time elapsed between this fact's last review.

Returns a new object (like `prior`) describing the posterior distribution of
recall probability at `tback` time after review.

If `rebalance` is True, the new object represents the updated recall
probability at *the halflife*, i,e., `tback` such that the expected
recall probability is is 0.5. This is the default behavior.

Performance-sensitive users might consider disabling rebalancing. In that
case, they may pass in the `tback` that the returned model should correspond
to. If none is provided, the returned model represets recall at the same time
as the input model.

N.B. This function is tested for numerical stability for small `total < 5`. It
may be unstable for much larger `total`.

N.B.2. This function may throw an assertion error upon numerical instability.
This can happen if the algorithm is *extremely* surprised by a result; for
example, if `successes=0` and `total=5` (complete failure) when `tnow` is very
small compared to the halflife encoded in `prior`. Calling functions are asked
to call this inside a try-except block and to handle any possible
`AssertionError`s in a manner consistent with user expectations, for example,
by faking a more reasonable `tnow`. Please open an issue if you encounter such
exceptions for cases that you think are reasonable.

<a name="ebisu.ebisu.modelToPercentileDecay"></a>
#### modelToPercentileDecay

```python
modelToPercentileDecay(model, percentile=0.5)
```

When will memory decay to a given percentile? 🏀

Given a memory `model` of the kind consumed by `predictRecall`,
etc., and optionally a `percentile` (defaults to 0.5, the
half-life), find the time it takes for memory to decay to
`percentile`.

<a name="ebisu.ebisu.rescaleHalflife"></a>
#### rescaleHalflife

```python
rescaleHalflife(prior, scale=1.)
```

Given any model, return a new model with the original's halflife scaled.
Use this function to adjust the halflife of a model.

Perhaps you want to see this flashcard far less, because you *really* know it.
`newModel = rescaleHalflife(model, 5)` to shift its memory model out to five
times the old halflife.

Or if there's a flashcard that suddenly you want to review more frequently,
perhaps because you've recently learned a confuser flashcard that interferes
with your memory of the first, `newModel = rescaleHalflife(model, 0.1)` will
reduce its halflife by a factor of one-tenth.

Useful tip: the returned model will have matching α = β, where `alpha, beta,
newHalflife = newModel`. This happens because we first find the old model's
halflife, then we time-shift its probability density to that halflife. The
halflife is the time when recall probability is 0.5, which implies α = β.
That is the distribution this function returns, except at the *scaled*
halflife.

<a name="ebisu.ebisu.defaultModel"></a>
#### defaultModel

```python
defaultModel(t, alpha=3.0, beta=None)
```

Convert recall probability prior's raw parameters into a model object. 🍗

`t` is your guess as to the half-life of any given fact, in units that you
must be consistent with throughout your use of Ebisu.

`alpha` and `beta` are the parameters of the Beta distribution that describe
your beliefs about the recall probability of a fact `t` time units after that
fact has been studied/reviewed/quizzed. If they are the same, `t` is a true
half-life, and this is a recommended way to create a default model for all
newly-learned facts. If `beta` is omitted, it is taken to be the same as
`alpha`.

