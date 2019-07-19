# Changes to Ebisu

## 1.0.0
**Breaking changes:**
- `predictRecall` returns log-probabilities, which are numbers between -∞ and 0 (log(0) being -∞ and log(1) being 0) by default, as a computational speedup. The returned values can still be sorted, and the lowest value corresponds to the lowest recall probability. Use `exact=True` to get true probabilities (at the cost of an `exp` function evaluation).
- The name of the half-life function is now `modelToPercentileDecay` and has a new API.

[Robert Kern's discovery](https://github.com/fasiha/ebisu/issues/5) that time-traveling Beta random variables through Ebbinghaus’ exponential decay function transform into GB1 random variables, which have analytic moments, was a major breakthrough. His contribution to this update, in code and ideas and time, cannot be overstated.

With the GB1 mathematical infrastructure, I was able to completely rethink the update step. Both passing and failing a quiz yield exact analytical moments of the posterior over any time horizon, not just when the test was taken. These are fit to a Beta at the very last minute. There is also a rebalancing step (which Robert foreshadowed in the GitHub [issue](https://github.com/fasiha/ebisu/issues/5) above as a “telescoping” posterior), wherein if one of the Beta’s parameters is large compared to the other, the update is rerun at the approximate half-life of the original unbalanced posterior fit.

All of these changes are transparent to the user, who will just see more accurate behavior in extreme over- and under-reviewing.

## 0.5.6
This version was tested by a couple of users of the Curtiz app (including the developer of Ebisu). It's `updateRecall` function returned models at the test time.