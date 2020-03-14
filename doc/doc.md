
# ebisu.ebisu Module


## Functions

##### `defaultModel(t, alpha=3.0, beta=None)` 

> Convert recall probability prior's raw parameters into a model object. 🍗
> 
>   `t` is your guess as to the half-life of any given fact, in units that you
>   must be consistent with throughout your use of Ebisu.
> 
>   `alpha` and `beta` are the parameters of the Beta distribution that describe
>   your beliefs about the recall probability of a fact `t` time units after that
>   fact has been studied/reviewed/quizzed. If they are the same, `t` is a true
>   half-life, and this is a recommended way to create a default model for all
>   newly-learned facts. If `beta` is omitted, it is taken to be the same as
>   `alpha`.



##### `modelToPercentileDecay(model, percentile=0.5, coarse=False)` 

> When will memory decay to a given percentile? 🏀
> 
>   Given a memory `model` of the kind consumed by `predictRecall`,
>   etc., and optionally a `percentile` (defaults to 0.5, the
>   half-life), find the time it takes for memory to decay to
>   `percentile`. If `coarse`, the returned time (in the same units as
>   `model`) is approximate.



##### `predictRecall(prior, tnow, exact=False)` 

> Expected recall probability now, given a prior distribution on it. 🍏
> 
>   `prior` is a tuple representing the prior distribution on recall probability
>   after a specific unit of time has elapsed since this fact's last review.
>   Specifically,  it's a 3-tuple, `(alpha, beta, t)` where `alpha` and `beta`
>   parameterize a Beta distribution that is the prior on recall probability at
>   time `t`.
> 
>   `tnow` is the *actual* time elapsed since this fact's most recent review.
> 
>   Optional keyword parameter `exact` makes the return value a probability,
>   specifically, the expected recall probability `tnow` after the last review: a
>   number between 0 and 1. If `exact` is false (the default), some calculations
>   are skipped and the return value won't be a probability, but can still be
>   compared against other values returned by this function. That is, if
> 
>   > predictRecall(prior1, tnow1, exact=True) < predictRecall(prior2, tnow2, exact=True)
> 
>   then it is guaranteed that
> 
>   > predictRecall(prior1, tnow1, exact=False) < predictRecall(prior2, tnow2, exact=False)
> 
>   The default is set to false for computational efficiency.
> 
>   See README for derivation.



##### `updateRecall(prior, success, total, tnow, rebalance=True, tback=None)` 

> Update a prior on recall probability with a quiz result and time. 🍌
> 
>   `prior` is same as in `ebisu.predictRecall`'s arguments: an object
>   representing a prior distribution on recall probability at some specific time
>   after a fact's most recent review.
> 
>   `success` is an integer representing the number of times the item was 
>   successfully recalled.
>
>   `total` is an integer representing the number ofimes the item was presented
>   for review. 
>
>   > `success` and `total` allow you to score review sessions where
>   > the same item was presented multiple times (and the student successfully
>   > recalled the item 1 out of 4 times, for example).
>
>   `tnow` is the time elapsed between this fact's last review and the review
>   being used to update.
> 
>   (The keyword arguments `rebalance` and `tback` are intended for internal use.)
> 
>   Returns a new object (like `prior`) describing the posterior distribution of
>   recall probability at `tback` (which is an optional input, defaults to `tnow`).


