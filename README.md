# Ebisu, version 3: the Ensemble, release candidate 1

## Intro

This is a release candidate for Ebisu v3. For v2, see the docs at https://fasiha.github.io/ebisu/. This document assumes you know what Ebisu generally is, so please refer to that for the full details, but in a nutshell, it's a library that helps flashcard app authors schedule quizzes in a probabilistically-sound Bayesian approach.

Version 3 fixes a long-standing problem with how previous versions modeled human memory: see https://github.com/fasiha/ebisu/issues/43 but in another nutshell, Ebisu v2 modeled your memory of a flashcard as a _fixed but unknown_ quantity: as you reviewed that flashcard several times, getting it right or wrong, the model's estimate of your memory's strength fluctuated (so Ebisu's predictions of your recall probability increased the more you got it right, and decreased otherwise) but obviously in the real world, the act of reviewing a flashcard actually _changes_, strengthens, your memory of it. This modeling flaw led Ebisu v2 to dramatically pessimistic predictions of recall probability.

Ebisu version 3 proposes a straightforward statistically-motivated extension to v2 that fixes this problem (described in https://github.com/fasiha/ebisu/issues/62). Whereas v2 posited a _single_ Beta random variable to model memory, v3 posits an _ensemble_ of a few Beta random variables, with decreasing weights. This allows v3 to have much more reasonable predictions of instantaneous recall probability and enables quiz app authors to use strategies like, "schedule a quiz when the probability of recall drops to 75%" (instead of my preferred approach, which is, "don't schedule, just quiz on the flashcard most in danger of being forgotten _right now_").

## Setup

[Ebisu v2](https://fasiha.github.io/ebisu/) has detailed docs, recipes, and mathematical analyses but for the sake of getting this release candidate out, I'm foregoing documentation and giving barebones instructions for my beta testers—thank you!

To install this release candidate:

```sh
python -m pip install "ebisu>=3rc"
```

## Usage

```py
import ebisu

# create an Ebisu model when the student has learned this flashcard
model = ebisu.initModel(
    firstHalflife=10, lastHalflife=10e3, firstWeight=0.9, numAtoms=5, initialAlphaBeta=2.0)

# at some point later, ask Ebisu for this flashcard's recall probability
timeSinceLastReview = 20
probabilityRecall = ebisu.predictRecall(model, timeSinceLastReview)
print(probabilityRecall)
# this is in the log-domain (for speed). You can ask for the normal (linear) probability
print(ebisu.predictRecall(model, timeSinceLastReview, logDomain=False))

# administer a quiz, then update the model, overwriting the old one
timeSinceLastReview = 20.1
model = ebisu.updateRecall(model, successes=1, total=1, elapsedTime=timeSinceLastReview)

# that's a binary quiz. You can also do a binomial quiz:
model = ebisu.updateRecall(model, successes=1, total=2, elapsedTime=timeSinceLastReview)

# you can also do fuzzy-binary quizzes, see Ebisu v2 docs
model = ebisu.updateRecall(model, successes=0.85, total=1, elapsedTime=timeSinceLastReview, q0=0.1)

# how long do we expect it to take for recall to drop to 50%?
print(ebisu.modelToPercentileDecay(model, 0.5))

# how long till recall drops to 80%?
print(ebisu.modelToPercentileDecay(model, 0.8))

# sometimes the model is just too hard or too easy. There's an ad hoc backdoor to rescaling it:
easierModel = ebisu.rescaleHalflife(model, 0.25)  # new halflife = 0.25 * old halflife
harderModel = ebisu.rescaleHalflife(model, 4)  # new halflife = 4 * old halflife
```

The biggest difference from v2 is that the new data model for memory have to be bigger, since we have a weighted ensemble of random variables instead of a single one: `initModel` needs your best guess as to its initial halflife and what the max halflife you want to track, in consistent units. Above, we initialized the model to have an initial halflife of 10 hours, so the first atom in our ensemble will be at 10 hours, but we'll also create four other atoms (for a total of `numAtoms=5`), logarithmically-spaced, till `lastHalflife=10e3` hours (1.1 years).

We initialize the first atom (10 hours) to have weight `firstWeight=0.9` and the rest of the atoms have logarithmically-decreasing weights. Finally, each atom is initialized with a Beta random variable whose `α = β = initialAlphaBeta = 2.0` (see Ebisu [v2 docs](https://fasiha.github.io/ebisu/) for what this means).

There's only one tiny change between v2 and v3's `predictRecall` API: now the keyword for switching between log-probability and normal probability is `logDomain=True` (the default) or `False`.

There is _no_ change to the API for
- `updateRecall`,
- `rescaleHalflife`, and
- `modelToPercentileDecay` (to calculate time till `predictRecall` drops to some value).

For details on the three types of quizzes mentioned above (binary, binomial, and fuzzy-noisy binary), see Ebisu [v2 docs](https://fasiha.github.io/ebisu/).

## Dev

### Tests

To run tests, first install this package locally:

```sh
python -m pip install -e .
```

(The `-e` means this install is "editable", so changes here will be automatically updated in the installed location.)

Then simply run

```sh
pytest
```
