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