
# ebisu.ebisu Module


## Functions

##### `defaultModel(t, alpha=4.0, beta=None)` 

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



##### `predictRecall(prior, tnow)` 

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
>   Returns the expectation of the recall probability `tnow` after last review, a
>   float between 0 and 1.
> 
>   See documentation for derivation.



##### `predictRecallVar(prior, tnow)` 

> Variance of recall probability now. 🍋
> 
>   This function returns the variance of the distribution whose mean is given by
>   `ebisu.predictRecall`. See it's documentation for details.
> 
>   Returns a float.



##### `priorToHalflife(prior, percentile=0.5, maxt=100, mint=0.001)` 

> Find the half-life corresponding to a time-based prior on recall. 🏀



##### `updateRecall(prior, result, tnow)` 

> Update a prior on recall probability with a quiz result and time. 🍌
> 
>   `prior` is same as for `ebisu.predictRecall` and `predictRecallVar`: an object
>   representing a prior distribution on recall probability at some specific time
>   after a fact's most recent review.
> 
>   `result` is truthy for a successful quiz, false-ish otherwise.
> 
>   `tnow` is the time elapsed between this fact's last review and the review
>   being used to update.
> 
>   Returns a new object (like `prior`) describing the posterior distribution of
>   recall probability at `tnow`.


