# Ebisu: intelligent quiz scheduling

- [Ebisu: intelligent quiz scheduling](#ebisu-intelligent-quiz-scheduling)
  - [Quick links](#quick-links)
  - [Introduction](#introduction)
  - [Install](#install)
  - [API Quickstart](#api-quickstart)
    - [Data model](#data-model)
    - [Predict recall probability](#predict-recall-probability)
    - [Quick-update halflife after a quiz](#quick-update-halflife-after-a-quiz)
    - [Fully-update halflife and boost](#fully-update-halflife-and-boost)
    - [Reset a model to a new halflife](#reset-a-model-to-a-new-halflife)
  - [Math](#math)
  - [Acknowledgments](#acknowledgments)

## Quick links
- [Literate document](https://fasiha.github.io/ebisu/)
- [GitHub repo](https://github.com/fasiha/ebisu)
- [PyPI package](https://pypi.python.org/pypi/ebisu/)
- [Changelog](https://github.com/fasiha/ebisu/blob/gh-pages/CHANGELOG.md)
- [Contact](https://fasiha.github.io/#contact)

## Introduction

Consider a student memorizing a set of facts.

- Which facts need reviewing?
- How does the student’s performance on a review change the fact’s future review schedule?

Ebisu is a public-domain library that answers these two questions. It is intended to be used by software developers writing quiz apps, and provides a simple API to deal with these two aspects of scheduling quizzes, centered on two functions:
- `predictRecall` gives the current recall probability for a given fact.
- `updateRecall` adjusts the belief about future recall probability given a quiz result.

Behind this simple API, Ebisu is using a simple yet powerful model of forgetting, a model that is founded on Bayesian statistics and exponential forgetting.

With this system, quiz applications can move away from “daily review piles” caused by less flexible scheduling algorithms. For instance, a student might have only five minutes to study today; an app using Ebisu can ensure that only the facts most in danger of being forgotten are reviewed.

Ebisu also enables apps to provide an infinite stream of quizzes for students who are cramming. Thus, Ebisu intelligently handles over-reviewing as well as under-reviewing.

This document contains both a detailed mathematical description of the underlying algorithm as well as the software API it exports. Separate implementations in other languages are detailed below.

The next section is a [Quickstart](#quickstart) guide to setup and usage. See this if you know you want to use Ebisu in your app.

Then in the [How It Works](#how-it-works) section, I contrast Ebisu to other scheduling algorithms and describe, non-technically, why you should use it.

Then there’s a long [Math](#the-math) section that details Ebisu’s algorithm mathematically. If you like Gamma-distributed random variables, importance sampling, and maximum likelihood, this is for you.

> Nerdy details in a nutshell: Ebisu begins by you positing [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution) priors on (1) the memory halflife of a newly-learned fact, as well as (2) a boost factor that governs how quickly that memory strengthens after each successful review. This allows you to easily find the facts that are most in danger of forgetting, via simple arithmetic. Next, a future *quiz* is modeled as a [binomial](https://en.wikipedia.org/wiki/Binomial_distribution) or noisy Bernoulli trial whose underlying probability is an exponential function of the halflife and the time elapsed since the fact was last reviewed. Ebisu then does a Bayesian update of the halflife, either using the boost as a static factor or jointly updating both halflife and boost. The static update is quite fast, involving an evaluation of the [Gamma function](https://en.wikipedia.org/wiki/Gamma_function) and the [Bessel function of the second kind](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kve.html). The joint update more expensive and involves fitting the bivariate posterior to a bivariate Gamma distribution followed by importance sampling.
 
Finally, in the [Source Code](#source-code) section, we describe the software testing done to validate the math, including comparing Ebisu’s updates to
1. vanilla Monte Carlo,
2. numerical integration, and
3. MCMC (Markov-chain Monte Carlo) via [Stan](https://mc-stan.org).

A quick note on history—more information is in the [Changelog](https://github.com/fasiha/ebisu/blob/gh-pages/CHANGELOG.md). This document discusses Ebisu v3. Versions 2 and before used a very different model that was simpler but failed to handle the strenghening of memory that accompanied quizzes. If you used Ebisu <=2 and are confused, see the [Changelog](https://github.com/fasiha/ebisu/blob/gh-pages/CHANGELOG.md) for a migration guide.

## Install
```sh
python -m pip install ebisu
```

## API Quickstart

### Data model
```py
def initModel(
    initHlMean: float,
    boostMean: float,
    initHlStd: Optional[float] = None,
    boostStd: Optional[float] = None,
    now: Optional[float] = None,
) -> Model
```
For each fact in your quiz app, create an Ebisu `Model` via `ebisu.initModel`. You give this function (a) the mean and (b) optionally the standard deviation of
1. this fact’s initial halflife in hours. This is your guess for how many hours it will take for the student’s recall probability to weaken to 50%. Anki starts out with 24 hours, Memrise with 4 hours; sometimes I use 0.25 (fifteen minutes). And
2. the boost factor that applies to each successful quiz. A good default might be 2. For facts that the student has created a good mnemonic for or has a strong mental model for, a good number might be 3 or 4. For facts where you expect weaker memory strenghening, maybe 1.5.

`now` is milliseconds since the Unix epoch (midnight UTC, Jan 1, 1970). Provide this to customize when the student learned this fact, otherwise Ebisu will use the current time.

Tip: Ebisu is pretty smart about boosting 😇 if you pass a quiz right after you study it, it’s obviously not going to boost the halflife by 4. You can take “boost” to mean “max boost” that Ebisu will apply to successful quizzes. More details in the math section below.

Tip: feel free to be generous with standard deviations if you don't have strong feelings about these values. See the math section below for details but you’re free to set the standard deviation up to `0.7 * mean` (technically `sqrt(0.5) * mean`). For the sake of providing a default, we’ll set the standard deviation to *half* the mean if you omit this.

You can serialize this `Model` with the `to_json` method provided by [Dataclasses-JSON](https://github.com/lidatong/dataclasses-json), which also provides its complement, `from_json`. Therefore, this will work:
```py
ebisu.Model.from_json(ebisu.initModel(24, 6, 2, 1).to_json())
```

It's expected that apps using Ebisu will save the serialized JSON to a database. The model contains all historic quiz information and numbers describing the probabilistic state. `Model.pred` has a two useful keys that will be useful for quiz apps directly:
- `Model.pred.lastEncounterMs`: a timestamp (milliseconds in Unix epoch) of when the student last encountered this quiz;
- `Model.pred.currentHalflifeHours`: the current estimate of this fact’s halflife.

### Predict recall probability
```py
def predictRecall(
    model: Model,
    now: Optional[float] = None,
    logDomain=True,
) -> float
```
This functions answers one of the core questions any flashcard app asks: what fact is most in danger of being forgotten? You can run this function against all Ebisu models, with an optional `now` (milliseconds in the Unix epoch), to get a log-probability of recall. A higher number implies more likely to recall; the lower the number, the more risk of of forgetting.

If you pass in `logDomain=False`, this function will call `exp` to convert log-probability (-∞ to 0) to actual probability (0 to 1). (This is not done by default because `exp`, a transcendental function, is actually expensive compared to arithmetic. No, I don’t have an explicit reference. Yes, profiling is important.)

**Nota bene** if you’re storing Ebisu models as JSON in SQL, you most likely do not need this function! The following snippet selects all columns and a new column, `scaledLogPredictRecall`, assuming a SQLite table called `mytable` with Ebisu models in a column called `model_json`:
```sql
SELECT *, 
       (JSON_EXTRACT(model_json, '$.pred.lastEncounterMs') - 
        strftime('%s','now') * 1000) /
       JSON_EXTRACT(model_json, '$.pred.currentHalflifeHours') AS scaledLogPredictRecall
FROM mytable
```
For reference, `scaledLogPredictRecall * math.log(2) / 3600e3` matches the ouput of `predictRecall`: `log(2)` to convert `exp` to `exp2`, and `3600e3` is milliseconds per hour.

### Quick-update halflife after a quiz
```py
def updateRecall(
    model: Model,
    successes: Union[float, int],
    total: int = 1,
    q0: Optional[float] = None,
    left=0.3,
    right=1.0,
    now: Optional[float] = None,
) -> Model
```
The other really important question flashcard apps ask is: “I've done a quiz, now what?” Ebisu has a two-step process to handle quiz results:
1. After each quiz, call `updateRecall` to get a new model, and save that to the database. This does a quick Bayesian update on just the halflife using just this quiz, assuming the boost to be fixed.
2. Once in a while, call the next function described below, `updateRecallHistory`, to perform a joint Bayesian update on both halflife *and* boost, incorporating *all* quizzes.

Via `updateRecall`, Ebisu supports **two** distinct kinds of quizzes:
- with `total=1` you get *noisy-binary* (or *soft-binary*) quizzes where `0 <= successes <= 1` can be a float.
- With `total>1` you get *binomial* quizzes, meaning out of a `total` number of points the student could get, she got `successes` (must be integer).

Example 1. For the bog-standard flashcard review, where you show the student a flashcard and they get it right (or wrong), you can pass in `successes=1` (or `successes=0`), and use the default `total=1`.

Example 2. For a Duolingo-style review, where you review the same fact multiple times in a single short quiz session, you provide the number of `successes` and `total>1` (the number of points received versus the maximum number of points, both integers).

Example 3. For more complex apps, where you have deep probabilistic insight into the student’s performance, you can use the noisy-binary by passing in `total=1` and `0 <= successes <= 1`. In the noisy-binary model, we assume the existence of a “real” binary quiz result which was scrambled by going through a noisy channel such that flips the “real” quiz result with some probability.
- `max(successes, 1 - successes)` is `Probability(observed pass | real quiz pass)` and
- `q0` is `Probability(observed pass | real quiz failed)`, defaults to the complement of the above (`1 - max(successes, 1 - successes)`).

A good example of a use for this quiz type: your app is a foreign language reader app and you know the student read a word and they did *not* ask for its definition. You don’t know that the student would have gotten it right if you’d *actually* prompted them for the definition of the word, so you don’t want to treat this as a normal “successful” quiz. So you can say
- `Probability(did not ask for definition | they know the word) = successes = 1.0`, i.e., if they know the word, they would never ask for the definition, but
- `Probability(did not ask for definition | they forgot the word) = q0 = 0.1`: if they actually had forgotten the word, there’s a low but non-zero chance of observing the same behavior (didn’t ask for the definition).

The two keyword arguments `left` and `right` let you customize a key feature of Ebisu's boosting mechanism. This is discussed more in the math section below.

As with other functions, `updateRecall` also accepts `now`, milliseconds since the Unix epoch.

### Fully-update halflife and boost
```py
def updateRecallHistory(
    model: Model,
    left=0.3,
    right=1.0,
    size=10_000,
    likelihoodFitWeight=0.9,
    likelihoodFitPower=2,
    likelihoodFitSize=600,
) -> Model
```
As mentioned above, flashcard apps are expected to call `updateRecall` after each quiz to quickly evolve the probability distribution around the memory halflife while holding the boost as fixed (by collapsing the boost’s probability distribution to its mean), using just a single quiz.

In contrast, this function `updateRecallHistory` will use all quizzes for this fact to jointly update an Ebisu model’s probability distributions for halflife *and* boost. The specifics are detailed in the math section below, but this is a computationally-intensive operation (matrix least-squares and importance sampling), taking between 100 milliseconds for five quizzes to ~700 milliseconds for twenty quizzes, on an old Mac laptop.

You could of course run `updateRecallHistory` after every quiz (call `updateRecall` and immediately call `updateRecallHistory`, and save its output to your app’s database).

Or you could run `updateRecallHistory` only once a day (after a quiz has been added to a model).

Or you could run `updateRecallHistory` every five quizzes.

Or some combination thereof, because after several quizzes, the initial priors you placed on the boost and halflife in `initModel` will become solidified, and `updateRecallHistory` will change these probability distributions less and less.

### Reset a model to a new halflife
```py
def resetHalflife(
    model: Model,
    initHlMean: float,
    initHlStd: Optional[float] = None,
    now: Optional[float] = None,
) -> Model
```
As mentioned above, once you have ten or twenty quizzes, each additional quiz result doesn’t make much of a difference in the probabilistic beliefs Ebisu has about that fact’s halflife and boost. Usually, this is a good thing! But it can also mean that when a fact becomes meaningfully easier (the student has internalized it) or harder (the student has learned a confuser fact, that interferes with the first), the Bayesian framework makes it hard to throw away the mass of old data and adapt to new information.

This function is a backdoor around that. It keeps all the old quiz data but reinitializes the halflife with a new mean and standard deviation (akin to `initModel`, though note how this function `resetHalflife` doesn’t change its belief about boost). Each new quiz can have a big impact on the probabilistic belief about the halflife. 

> You might not need this function. If it turns out you just initialized the model with a bad mean/standard deviation for halfife and boost (perhaps they were too narrow and the new quiz data cannot overcome the prior?), you can just reset `Model.prob.initHlPrior` and `Model.prob.boostPrior` and rerun `updateModelHistory` to see if thath elps.

That's it. Five functions in the API.

## Math
(Forthcoming.)

## Acknowledgments

A huge thank you to [bug reporters and math experts](https://github.com/fasiha/ebisu/issues?utf8=%E2%9C%93&q=is%3Aissue) and [contributors](https://github.com/fasiha/ebisu/graphs/contributors)!

Many thanks to [mxwsn and commenters](https://stats.stackexchange.com/q/273221/31187) as well as [jth](https://stats.stackexchange.com/q/272834/31187) for their advice and patience with my statistical incompetence.

Many thanks also to Drew Benedetti for reviewing this manuscript.

John Otander’s [Modest CSS](http://markdowncss.github.io/modest/) is used to style the Markdown output.
