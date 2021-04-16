# Changes to Ebisu

## 2.1.0: soft-binary quizzes and halflife rescaling

`updateRecall` can now take *floating point* quizzes between 0 and 1 (inclusive) as `successes`, to handle the case when your quiz isn’t quite correct or wasn’t fully incorrect. Varying this number between 0 and 1 will smoothly vary the halflife of the updated model. Under the hood, there's a noisy-Bernoulli statistical model.

A new function has been added to the API, `rescaleHalflife`, for those cases when the halflife of a flashcard is just wrong and you need to multiply it by ten (so you see it less often) or divide it by two (so you see it *more* often).

A behavioral change: `updateRecall` will by default rebalance models, so that updated models will have `α=β` (to within machine precision) and `t` will be the halflife. This does have a performance impact, so I may have to flip the default to *not* always rebalance in a future release if this turns out to be problematic.

Closes long-standing issues [#23](https://github.com/fasiha/ebisu/issues/23) and [#31](https://github.com/fasiha/ebisu/issues/31)—thank you to all participants who weighed in, offered advice, and waited patiently.

## 2.0.0: Bernoulli to binomial quizzes
The API for `updateRecall` has changed because `boolean` results don't make sense for quiz apps that have a sense of "review sessions" in which the same flashcard can be reviewed more than one time, e.g., if a review session consists of conjugating the same verb twice. Therefore, `updateRecall` accepts two integers:
- `successes`, the number of times the user correctly produced the memory encoded in this flashcard, out of
- `total` number of times it was presented to the user.

The old behavior can be recovered by setting `total=1` and `successes=1` upon success and 0 upon failure.

The memory models from previous versions remain fully-compatible with this update.

While this new feature allows more freedom in desining quiz applications, it does open up the possibility of numerical instability when the function receives a very surprising input. Please wrap calls to `updateRecall` in a `try` block to gracefully handle this possibility, and get in touch in case it happens to you a lot.

## 1.0.0
**Breaking changes:**
- `predictRecall` returns log-probabilities, which are numbers between -∞ and 0 (log(0) being -∞ and log(1) being 0) by default, as a computational speedup. The returned values can still be sorted, and the lowest value corresponds to the lowest recall probability. Use `exact=True` to get true probabilities (at the cost of an `exp` function evaluation).
- The name of the half-life function is now `modelToPercentileDecay` and has a new API.

[Robert Kern's discovery](https://github.com/fasiha/ebisu/issues/5) that time-traveling Beta random variables through Ebbinghaus’ exponential decay function transform into GB1 random variables, which have analytic moments, was a major breakthrough. His contribution to this update, in code and ideas and time, cannot be overstated.

With the GB1 mathematical infrastructure, I was able to completely rethink the update step. Both passing and failing a quiz yield exact analytical moments of the posterior over any time horizon, not just when the test was taken. These are fit to a Beta at the very last minute. There is also a rebalancing step (which Robert foreshadowed in the GitHub [issue](https://github.com/fasiha/ebisu/issues/5) above as a “telescoping” posterior), wherein if one of the Beta’s parameters is large compared to the other, the update is rerun at the approximate half-life of the original unbalanced posterior fit.

All of these changes are transparent to the user, who will just see more accurate behavior in extreme over- and under-reviewing.

## 0.5.6
This version was tested by a couple of users of the Curtiz app (including the developer of Ebisu). Its `updateRecall` function returned models at the test time.
