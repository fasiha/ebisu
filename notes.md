# Sun Jun  4 17:38:24 PDT 2023
Thinking about
- I am confident I can remove all the models in `ebisu/models.py` and switch to a v2-style "model" of just `list[tuple[float, float, float]]` where the tuple is `α, β, log2weight`.
- But that reminded me of a question I need to answer: how can we do `rescaleHalflife` in v3 ensemble?

Rescaling the halflife in an ensemble *might* be a bit easier if we kept history like the current model: if we do a quiz and the student says "rescale this by 2" (or 0.25 or whatever), we could go back to the previous update and apply a doctored result that… somehow makes the current probability of recall 2x higher (or 4x lower or whatever)?

What would that quiz update look like. Recall that the particle filter weight update is just scaling the old weight by this atom's probability of seeing the quiz. Different atoms will scale their weights differently.

Most times the weights of our atoms will look like a hump, low then high then low. We want to skew those a bit so the hump moves to the right. Is there something in the literature on Dirichlet distributions that tells us how to scale a simplex like this? We want to keep the weights summing to one, so what constraints are there on how we scale each component (scale down the left-most components and scale up the right-most ones)?

Do we want to incorporate halflife here?

One advantage of keeping the history of quizzes in the model is that we could separate the real quiz from the adjustment applied for rescaling.

# Wed Jun 14 21:32:38 PDT 2023
Thinking about `rescaleHalflife`. I like the `rescaleHalflife` function in v2's API. This is where, you finish a quiz, and you just want to tell Ebisu "you are wrong about this model, just rescale this model's halflife by X" where X is some number greater than 0. If you thought the quiz was too easy, then you can set X=2 (or 10 if you thought it was way too easy). If you thought it was too hard, you set X=0.5 or 0.1. It's trivial to do this in v2 (a single atom) but not obvious how to best do this in an ensemble.

# Fri Jun 16 21:45:32 PDT 2023
In writing an email explaining the `power=20` I noticed that `power` and weights both do the same thing but in opposite directions: the weights reduce the raw recall probability `2**(-t/h)`, but the power "increases" it, and the power is more "final": consider
```py
def powermean(xs, ws, p: float):
  return (sum(w * x**p for w, x in zip(ws, xs)) / sum(ws))**(1 / p)

atomicPrecalls = [1.7e-21, 0.00018, 0.22, 0.77, 0.95]
atomWeights = [0.8, 0.17, 0.028, 0.0046, 0.00077]
print([powermean(atomicPrecalls, atomWeights, x) for x in [1, 10, 100, 1000]])
# Result: 0.01, 0.49, 0.89, 0.95
```
Here, note that the final weight on the longest halflife is quite low but as `power` increases, the effect of that weight is basically canceled out and the final result tends towards the raw recall of that longest halflife atom, 95%. 

This makes `power` very much an "important to tune" parameter and I very much dislike having two knobs that basically fight each other like this. I am seriously considering setting the default `power=1`, i.e., using the arithmetic mean. Even tweaking the `w1` to yield the best log-likelihood, this results in often much worse log-likelihood scores versus `power=20`. However, looking at the raw predictions for each quiz in each flashcard (`ankiCompare.py`), two benefits of `power=1` emerge:
1. its 80% interval are much more moderate than the `power=20` examples, i.e., for a weak card its final 80% interval is higher and for strong cards its 80% interval is lower than the `power=20` case, and much more reasonable.
2. Also for a weak card like 1300038030838 where `power=1` has log-likelihood of -22.833 versus -17.031 of `power=20`, you can see that the `power=1` predictor was pessimistic in exactly the right times: on successes inbetween failures.

So I think `power=1` is a pretty good choice.

# Sat Jun 17 22:00:24 PDT 2023
Going back to `rescaleHalflife`. A thought experiment: suppose you want to drastically increase the halflife by a specific amount, you think the model is too pessimistic. I'm wondering if there's a difference in impact on the weights update between
- binominal 3 out of 3 at the current/future time versus
- 1 out of 1 (normal binary) at some even farther future time

as a way to fake a update that results in the halflife increasing say 5x.

> Sidebar: I was wondering if we still wanted `rescaleHalflife` in the API, maybe just binomial quizzes could substitute? No I don't think so because the student has beliefs about the halflife and a 3/3 or 4/4 update might result in a much greater halflife boost than they want.

The new weight is just the old weight plus log-likelihood of the observation. Assuming we found the two times that matched the two cases above, both yielding a 5x increase in halflife, that would increment all weights the same, but what is the relation between those hypothetical quizs' log-likelihoods? Are they the same, one greater than the other?

Answer: they're equal! I.e., if you want to increase the time-to80-percentile by 5x, a 3-out-of-3 result needs to be at 752 hours and a 1-out-of-1 in 2256 hours, and both have the exact same log-likelihoods under all atoms. (Probably the same for 0.5x.)

```py
from scipy.optimize import minimize_scalar
import numpy as np
import ebisu

m = ebisu.initModel(halflife=100, now=0, power=20, n=5, stdScale=1, w1=.5)

hlScale = 5

percentile = 0.8
initDecay = ebisu.hoursForRecallDecay(m, percentile)

updateKwargs = dict(updateThreshold=0.99, weightThreshold=0.05)
solBinominal = minimize_scalar(lambda h: abs(initDecay * hlScale - ebisu.hoursForRecallDecay(
    ebisu.updateRecall(m, 3, 3, now=h * 3600e3, **updateKwargs), percentile)))
solBinary = minimize_scalar(lambda h: abs(initDecay * hlScale - ebisu.hoursForRecallDecay(
    ebisu.updateRecall(m, 1, 1, now=h * 3600e3, **updateKwargs), percentile)))


def successToLoglik(success: int, t: float, h: float) -> float:
  return ebisu.resultToLogProbability(ebisu.BinomialResult(success, success, t), 2**(-t / (h)))


lls = [[successToLoglik(3, solBinominal.x, a / b) for (a, b) in m.pred.halflifeGammas],
       [successToLoglik(1, solBinary.x, a / b) for (a, b) in m.pred.halflifeGammas]]
print(np.array(lls[0]) / np.array(lls[1]))
# prints [1.00000001 1.00000001 1.00000001 1.00000001 1.00000001]
```

Note that we don't actually want to create a quiz like this. We just do this to find the new weights.

> We discovered while working on the boost-Gamma model that it's not sufficient to just adjust the initial priors and re-run all the quizzes since then. That's not a reliable rescaling. We really need a hard break: "this model right now is too pessimistic/optimistic, by THIS amount". 

So the proposal is, the student has completed a quiz and is giving us a result and a rescale factor. There's a risk that the result is inflated and overlaps with the rescale factor (i.e., we don't want to student to say 2/2 and scale the halflife by 5x, because that's a bigger change than a 1/1). So we ignore the result. The result shouldn't factor in post-hoc analyses that we do (like the log-likelihood analysis in `ankiCompare.py`). We can create a NoisyBinary quiz type with q1=q0=0.5 and a "rescale" field to indicate the value. When `updateRecall` sees this it knows to apply the rescale logic instead of the normal update.

> Sidebar. We use binomial quizzes to map Anki "easy" to 2-of-2 but I don't want to encourage this in quiz apps. I think rescaling is better.

So for whatever reason your quiz app decided to show the student this quiz at a certain time (maybe because it was "due", maybe because it had low recall probability). At that time the student says "whatever probability you think, it's not right, scale it by this amount". That scale is phrased as halflife. You give Ebisu the old model, Ebisu will do what? In the Python script above, it rescales the 80-percentile-life by 5x. The result is quite different if it scaled the halflife (50-percentile-life). 

# Sun Jun 18 17:30:24 PDT 2023
Here's the solution: there's two ways to rescale.
1. pass `rescale: float > 0` to `updateRecall`. This find an imaginary quiz that moves the old model to the current probability of recall,

# Sat Jul 22 21:21:15 PDT 2023
I've been thinking about a small simplification of the ebisu v3 mixture-of-Gammas: a mixture of Betas! Just reuse all the same code of Ebisu v2. That's now in `v3-ensemble` branch (`aa5f853`). From a log-likelihood perspective, mixture-of-Betas is competitive with mixture-of-Gammas. On hard quizzes, with a lot of initial failures that then the student properly learns eventually and has long-duration successful quizzes for, the mixture-of-Betas is slower to catch up than a mixture-of-Gammas but again as we discovered when deciding the fix `power=1`, this is not necessarily a bad thing to be conservative—and in fact the mixture-of-Betas outperforms its opponent in log-likelihood terms, -20.1 vs -23.4.

Scratch that. If you configure the initial halflife, the first atom's weight, and the update weight thresholds appropriately, the Betas and Gammas model line up very similarly, in terms of log-likelihood!

With `weightThreshold=0.05`:
```
loglikFinal=[-5.347  , -5.4]      , card.key=1300038031922
loglikFinal=[-23.401 , -23.407]   , card.key=1300038030838
loglikFinal=[-15.33  , -15.405]   , card.key=1300038030700
loglikFinal=[-14.575 , -14.333]   , card.key=1300038030475
loglikFinal=[-12.989 , -12.747]   , card.key=1300038030479
loglikFinal=[-7.302  , -7.263]    , card.key=1300038030454
loglikFinal=[-3.392  , -3.274]    , card.key=1300038030426
```

With `weightThreshold=0.49`: a bit more variability:
```
loglikFinal=[-5.137  , -5.12]     , card.key=1300038031922
loglikFinal=[-19.393 , -20.32]    , card.key=1300038030838
loglikFinal=[-14.543 , -14.305]   , card.key=1300038030700
loglikFinal=[-13.113 , -13.859]   , card.key=1300038030475
loglikFinal=[-10.991 , -11.756]   , card.key=1300038030479
loglikFinal=[-6.354  , -6.676]    , card.key=1300038030454
loglikFinal=[-3.392  , -3.274]    , card.key=1300038030426
```

# Wed Jul 26 21:07:54 PDT 2023
Writing up notes about comparing Beta vs Gamma, why is the Gamma difference between exact and approx so terrible?


> I'm now investigating what happens if we use a mixture of Betas (the very very straightforward extension of v2)

Posted https://github.com/fasiha/ebisu/issues/62#issuecomment-1689268179

# Tue Aug 22 21:46:50 PDT 2023
Next question: do I lug around this big `Model` dataclass or do I stick with the v2 style of keeping my model really simple? List of Beta models `(α, β, t, weight)`?

## Improve initial weights

I don't know why I did this weird weight thing. This is simple and correct:
```py
w1=.9
n=4
sol = minimize_scalar(lambda x: abs(sum(w1 * (x)**i for i in range(n)) - 1), [0.3, 0.6])
print(w1 * sol.x**np.arange(n))
```

# Mon Sep 18 22:04:32 PDT 2023
## Rescaling
I'm thinking about how to move the high-weight atoms around to achieve a rescale request (scale the halflife by 0.5, 3x, etc.).