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