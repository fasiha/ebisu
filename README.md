# Ebisu, version 3: the Ensemble, release candidate 1

- [Ebisu, version 3: the Ensemble, release candidate 1](#ebisu-version-3-the-ensemble-release-candidate-1)
  - [Release candidate quick-intro](#release-candidate-quick-intro)
    - [Setup](#setup)
    - [Usage](#usage)
  - [Introduction](#introduction)
  - [Install](#install)
  - [API](#api)
    - [Prelude: import](#prelude-import)
    - [1. Create a model](#1-create-a-model)
    - [2. Predict recall](#2-predict-recall)
    - [3. Update recall](#3-update-recall)
      - [Simple binary quizzes](#simple-binary-quizzes)
      - [Binomial quizzes](#binomial-quizzes)
      - [Noisy-binary quizzes](#noisy-binary-quizzes)
    - [Other update parameters](#other-update-parameters)
    - [Bonus: time to recall decay](#bonus-time-to-recall-decay)
    - [Bonus: rescale halflife](#bonus-rescale-halflife)
  - [How it works](#how-it-works)
  - [Math](#math)
    - [Bernoulli quizzes](#bernoulli-quizzes)
    - [Moving Beta distributions through time](#moving-beta-distributions-through-time)
  - [Dev](#dev)
    - [Tests](#tests)
  - [Deploy to PyPI](#deploy-to-pypi)

## Release candidate quick-intro

This is a release candidate for Ebisu v3. For v2, see the docs at https://fasiha.github.io/ebisu/. This document assumes you know what Ebisu generally is, so please refer to that for the full details, but in a nutshell, it's a library that helps flashcard app authors schedule quizzes in a probabilistically-sound Bayesian approach.

Version 3 fixes a long-standing problem with how previous versions modeled human memory: see https://github.com/fasiha/ebisu/issues/43 but in another nutshell, Ebisu v2 modeled your memory of a flashcard as a _fixed but unknown_ quantity: as you reviewed that flashcard several times, getting it right or wrong, the model's estimate of your memory's strength fluctuated (so Ebisu's predictions of your recall probability increased the more you got it right, and decreased otherwise) but obviously in the real world, the act of reviewing a flashcard actually _changes_, strengthens, your memory of it. This modeling flaw led Ebisu v2 to dramatically pessimistic predictions of recall probability.

Ebisu version 3 proposes a straightforward statistically-motivated extension to v2 that fixes this problem (described in https://github.com/fasiha/ebisu/issues/62). Whereas v2 posited a _single_ Beta random variable to model memory, v3 posits an _ensemble_ of a few Beta random variables, with decreasing weights. This allows v3 to have much more reasonable predictions of instantaneous recall probability and enables quiz app authors to use strategies like, "schedule a quiz when the probability of recall drops to 75%" (instead of my preferred approach, which is, "don't schedule, just quiz on the flashcard most in danger of being forgotten _right now_").

### Setup

[Ebisu v2](https://fasiha.github.io/ebisu/) has detailed docs, recipes, and mathematical analyses but for the sake of getting this release candidate out, I'm foregoing documentation and giving barebones instructions for my beta testers—thank you!

To install this release candidate:

```sh
python -m pip install "ebisu>=3rc"
```

### Usage

```py
import ebisu

# create an Ebisu model when the student has learned this flashcard
model = ebisu.initModel(
    firstHalflife=10, lastHalflife=10e3, firstWeight=0.9, numAtoms=5, initialAlphaBeta=2.0)

# at some point later, ask Ebisu for this flashcard's recall probability
timeSinceLastReview = 20
probabilityRecall = ebisu.predictRecall(model, timeSinceLastReview)
print(probabilityRecall)

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

There's only one tiny change between v2 and v3's `predictRecall` API: there's no extra argument for whether you want the result as log-probability or normal probability. This function always returns normal (linear) probability.

There is _no_ change to the API for

- `updateRecall`,
- `rescaleHalflife`, and
- `modelToPercentileDecay` (to calculate time till `predictRecall` drops to some value).

For details on the three types of quizzes mentioned above (binary, binomial, and fuzzy-noisy binary), see Ebisu [v2 docs](https://fasiha.github.io/ebisu/).

## Introduction

Consider a student memorizing a set of facts.

- Which facts need to be reviewed?
- How does the student’s performance on a review change the fact’s future review schedule?

Ebisu is an open-source public-domain library that answers these two questions. It is intended to be used by software developers writing quiz apps, and provides a simple API to deal with these two aspects of scheduling quizzes, centered on two functions:

- `predictRecall` gives the recall probability for a given fact _right now_ or at any given timestamp.
- `updateRecall` adjusts the belief about _future_ recall given a quiz result.

Behind this simple API, Ebisu is using a simple yet powerful model of forgetting, a model that is founded on Bayesian statistics and series-of-exponentials (power law) forgetting. Because of this probabilistic foundation, Ebisu allows quiz applications to move away from “daily review piles” caused by less flexible scheduling algorithms. For instance, a student might have only five minutes to study today—an app using Ebisu can ensure that only the facts most in danger of being forgotten are presented for review. And since you can query every fact's recall probability at any time, Ebisu also enables apps to provide an infinite stream of quizzes for students who are cramming. Thus, Ebisu intelligently handles over-reviewing as well as under-reviewing. That's not all!

The probabilistic foundation also allows Ebisu to handle quite a rich variety of _quiz types_:

- of course you have your binary quizzes, i.e., pass/fail;
- you also have Duolingo-style quizzes where the student got X points out of a maximum of Y points (binomial quizzes);
- you can even customize the probability that the student “passed” the quiz even if they forgot the fact—this is handy for deweighting multiple-choice quizzes, or for reader apps where the readers can click on words they don’t know, or not.

In a nutshell, Ebisu has been able to support creative quiz apps with innovative review systems, not just simple pass/fail flashcards.

## Install

```sh
python -m pip install ebisu
```

(But to install a release candidate, use `python -m pip install "ebisu>=3rc"`)

## API

### Prelude: import

**Step 0.** `import ebisu`

### 1. Create a model

**Step 1.** Create an Ebisu `Model` for each flashcard when the student learns it:

```py
def initModel(
    firstHalflife: float,
    lastHalflife: Optional[float] = None,  # default will be `10_000 * firstHalflife`
    firstWeight: float = 0.9,
    numAtoms: int = 5,
    initialAlphaBeta: float = 2.0,
) -> BetaEnsemble
```

The only required argument is `firstHalflife`, your best guess as to how much time it'll take for this student's probability of recall for this fact to drop to 50%. Ebisu doesn't care about what time units you use so you have to be consistent: pick hours or days or whatever, and use the same unit everywhere.

> Ebisu is Bayesian, so it doesn't treat `firstHalflife` as "the truth", but rather as your probabilistic belief. More specifically, you are telling Ebisu that that after `firstHalflife` time units have elapsed, the student's memory will decay to a `Beta(α, β)` random variable, with both parameters, `α` and `β`, set to `initialAlphaBeta > 1`. Here's what that looks like visually for a few different `initialAlphaBeta`:
>
> ![Distribution of Beta random variables with parameters 1.25, 2, 5, and 10](./figures/betas.svg)
>
> This chart can be confusing if you're not used to thinking about probabilities on probabilities 😅! It shows that, no matter what you pick for `initialAlphaBeta`, the student's recall probability after one halflife is 50-50, since all four curves peak at 0.5. However, varying `initialAlphaBeta` encodes different beliefs about how _confident_ you are in your claim that this is the halflife. The default `initialAlphaBeta=2` gives you quite a lot of wiggle room: the student's probability of recall after one halflife might be as low as 20% or as high as 80%. However, setting a high `initialAlphaBeta` says "I'm _really_ confident that the halflife I tell you is really the halflife".

Morever, Ebisu creates not just one such Beta random variable but `numAtoms` of them, each with a halflife that logarithmically-decreases from `firstHalflife` to `lastHalflife`. (History note, this is the main innovation from Ebisu v2 to v3: more than one atom.) This is important because human memory is fat-tailed (more on this below in the [math](#math) section!): sometimes a fact that you learned just now you'll remember for years without review, and having an atom encoding a halflife of a year can capture this.

However, we obviously believe that the recall probability is much more likely to be governed by the first atom than the last. That's why each atom has a logarithmically-decreasing weight, with the first main atom assigned `firstWeight`. By adjusting this, you can tune how quickly or slowly later atoms respond to quizzes.

> So for example if I pick `firstHalflife=10` hours, by default the five atoms will have
>
> 1. Halflife 10 hours; weight 90%
> 1. Halflife 4 days and 4 hours; weight 9%
> 1. Halflife 1 month and 10 days; weight 0.9%
> 1. Halflife 1 year and 1 month; weight 0.09%
> 1. Halflife 11 years; weight 0.009%
>
> If you divide each row's halflife (or weight) by the previous row's, you'll find the same ratio. That's thanks to the logarithmic spacing.

The model returned by `initModel` is easily encoded into JSON. I recommend you include it in your database along with the timestamp it was learned.

### 2. Predict recall

So, a student has learned a few facts, you've called `initModel` each time and saved all models in a database.

Now, what is the probability of recall for those flashcards?

```py
def predictRecall(
    prior: BetaEnsemble,
    elapsedTime: float,
    logDomain=True,
) -> float
```

Just give each model to the above function along with how long it's been since the student last saw this flashcard, `elapsedTime`; be sure to use the same units that you used for `initModel`!

(By default, this function will return the log-probability, to save an exponential. Pass in `logDomain=False` to get the normal probability. Log-probability goes from -∞ to 0 whereas regular probability goes from 0 to 1.)

If the student wants to review, you can pick the flashcard with the lowest `predictRecall` and quiz them on that. (If your app schedules a review for when the recall drops below some threshold, see below 👇 for `modelToPercentileDecay`.)

Ebisu also provides `predictRecallApprox` with the same API as `predictRecall` that uses a fast approximation.

### 3. Update recall

Now you've quizzed the student on a flashcard and you want to update the the model with the quiz results.

```py
def updateRecall(
    prior: BetaEnsemble,
    successes: Union[float, int],
    total: int,
    elapsedTime: float,
    q0: Optional[float] = None,
    updateThreshold=0.99,
    weightThreshold=0.49,
) -> BetaEnsemble
```

There are two-ish distinct quiz types that Ebisu supports thanks to its Bayesian formulation, so this function has a lot of potential arguments.

#### Simple binary quizzes

On success, `updateRecall(model, 1, 1, elapsedTime)`.

On failure, `updateRecall(model, 0, 1, elapsedTime)`. That is, `successes` is 0 or 1 out of a `total=1` and `elapsedTime` in units consistent with `predictRecall` and `initModel`.

#### Binomial quizzes

Binary quizzes are a special case of binomial quizzes, where, in a single quiz session, the student got some points out of a max number of points. For example, `updateRecall(model, 2, 3, elapsedTime)` for when a student gets 2 points out of 3.

> Ebisu is well-tested for up to `total=5` points, beyond which the algorithm might become numerically stable.

#### Noisy-binary quizzes

In this case, `total=1` and `0 < successes < 1` is a _float_. See the [math](#math) section for more details but `successes` and the optional `q0` argument allow you to specify quizzes where you have doubts about the outcome you observed.

- `successes >= 0.5` implies the quiz was ostensibly a success
- `q1 = successes if successes >= 0.5 else 1 - successes` is the probability that the quiz was a success _assuming_ in reality the student knows the fact
- `q0` meanwhile is the probability that the quiz was a success when in fact the student has _forgotten_ this fact (by default `q0 = 1 - q1`)

This is useful for multiple choice quizzes where the student may have forgotten the answer but just guessed lucky, or for reading apps that observe the student _did not_ click on a word for a definition (they might not have needed the definition becuase they knew it, or maybe they forgot the word and didn't bother to click), and several other interesting niche applications.

> Note that the noisy-binary with default `q0` will smoothly interpolate between the binary success and binary failure mode as `successes` goes from 1.0 to 0.0.

> Note that one quiz mode we do _not_ directly support is Anki-style "easy, normal, hard" quizzes. An ad hoc way to do this might be to use a Binomial quiz, mapping
>
> - easy to `successes=2, total=2`,
> - normal to `successes=1, total=1`,
> - hard to `successes=1, total=2`?
>
> but you will want to experiment with this.

### Other update parameters

Two final `updateRecall` parameters we haven't talked about are:

- `updateThreshold` and
- `weightThreshold`.

Although the defaults are sane, advanced users may consider tweaking these values for their applications. And indeed, these are expected to be fixed for a given application (rather than varied card-to-card). These two parameters ensure that low-weight atoms with halflives in the far future don't get perturbed by high-impact quizzes. We dissect this in the [math](#math) section below.

### Bonus: time to recall decay

Given an Ebisu model, this function calculates how long it takes the model to decay to some `percentile`. When `percentile=0.5`, the returned value is the halflife. The returned value has same units as used everywhere else.

```py
def modelToPercentileDecay(model: BetaEnsemble, percentile=0.5) -> float
```

### Bonus: rescale halflife

Everyone has had the experience where, they do a quiz and the memory model was just wrong: the quiz was too hard or too easy and you want some simple way to just scale the whole model. This function is designed for this:

```py
def rescaleHalflife(model: BetaEnsemble, scale: float) -> BetaEnsemble
```

For a model that is far too easy (quiz happened far too quickly), call with `scale > 1` to rescale its halflife by that amount, e.g., `scale=2` will rescale each atom's halflife by 2.

For a model that's too hard (quiz was too delayed), call with `scale < 1`, e.g., `scale=0.25` to rescale each halflife to a quarter of its old value.

This function is offered as a backdoor—this recsaling is totally ad hoc amounts to just throwing away the Bayesian priors that you encoded in `initModel` and updated through successive `updateRecall`s and just shifting everything around by `scale`. This is sometimes necessary though! Perhaps the student has suddenly solidified their understanding of a fact, or perhaps they've learned some new fact that's interfering with this fact.

An alternative to using this function might be to reconsider the parameters you initialized the model with in `initModel`. Perhaps the initial halflife was wrong, perhaps its starting weight was bad. Consider picking a better initialization and rerunning `updateRecall` on all quizzes you've done for this flashcard. This is also somewhat improper since `initModel` is supposed to be your best guess _before_ you see any data, and going back to adjust your priors after you've seen the data is a big Bayesian no-no.

---

That's it, that's the API!

## How it works

There are many flashcard scheduling schemes, e.g.,

- [Anki](https://apps.ankiweb.net/), an open-source Python flashcard app (and a closed-source mobile app),
- the [SuperMemo](https://www.supermemo.com/help/smalg.htm) family of algorithms ([Anki’s](https://faqs.ankiweb.net/what-spaced-repetition-algorithm.html) is a derivative of SM-2),
- [Memrise.com](https://www.memrise.com), a closed-source webapp,
- [Duolingo](https://www.duolingo.com/) has published a [blog entry](http://making.duolingo.com/how-we-learn-how-you-learn) and a [conference paper/code repo](https://github.com/duolingo/halflife-regression) on their half-life regression technique,
- the Leitner and Pimsleur spacing schemes (also discussed in some length in Duolingo’s paper).
- Also worth noting is Michael Mozer’s team’s Bayesian multiscale models, specifically Mozer et al. (2009) and, by the same team, Lindsey et al. (2014) (see [bibliography](#bibliography)).

Memory research began with Hermann Ebbinghaus’ discovery of the [forgetting curve](https://en.wikipedia.org/w/index.php?title=Forgetting_curve&oldid=766120598#History), published in 1885, when he was thirty-five. He [memorized random](https://en.wikipedia.org/w/index.php?title=Hermann_Ebbinghaus&oldid=773908952#Research_on_memory) consonant–vowel–consonant trigrams (‘PED’, e.g.) and found, among other things, that his recall decayed logarithmically. More recent research has shown, apparently conclusively, that _forgetting_ follows a power law decay.

Anki and SuperMemo are extremely popular flashcard apps. They use carefully-tuned mechanical rules to schedule a fact’s future review immediately after its current review. The rules can get complicated—I wrote a little [field guide](https://gist.github.com/fasiha/31ce46c36371ff57fdbc1254af424174) to Anki’s, with links to the source code—since they are optimized to minimize daily review time while maximizing retention. However, because each fact has simply a date of next review, these algorithms do not gracefully accommodate over- or under-reviewing. Even when used as prescribed, they can schedule many facts for review on one day but few on others. (I must note that all three of these issues—over-reviewing (cramming), under-reviewing, and lumpy reviews—have well-supported solutions in Anki by tweaking the rules and third-party plugins.)

Duolingo’s half-life regression explicitly models the probability of you recalling a fact as an exponential, $2^{-Δ/h}$ where Δ is the time since your last review and $h$ is a _half-life_. In this model, your chances of passing a quiz after $h$ days is 50%, which drops to 25% after $2 h$ days, and so on. They estimate this half-life by combining your past performance and fact metadata in a large-scale machine learning technique called half-life regression (a variant of logistic regression or beta regression, more tuned to this forgetting curve). With each fact associated with a half-life, they can predict the likelihood of forgetting a fact if a quiz was given right now. The results of that quiz (for whichever fact was chosen to review) are used to update that fact’s half-life by re-running the machine learning process with the results from the latest quizzes.

The Mozer group’s algorithms (MCM (their 2009 paper) and DASH (their 2014 paper; see [bibliography](#bibliography))) also curve-fit a large quantity of quiz data to high-dimensional models, including, in DASH’s case, a hierarchical Bayesian model that takes into account inter-fact and inter-student variability.

Like Duolingo and Mozer et al., Ebisu explicitly tracks a forgetting curve for each flashcard, enabling it to sort a list of flashcards from most to least likely to be forgotten. However, Ebisu formulates the problem very differently—while memory is understood to decay according to a specific formula, Ebisu posits _probability distributions_ on the halflives governing this decay and uses quiz results to update its beliefs about those halflives in a fully Bayesian way. These updates, while a bit more computationally-burdensome than Anki’s scheduler, are much lighter-weight than Duolingo’s industrial-strength approach.

Being Bayesian grants Ebisu some really useful features:

1. Quiz app authors can encode quite specific beliefs about their students’ initial memory of a flashcard. If you have a reason to believe that vocabulary item A is much more easy to learn than item B, you can specify that exactly and quantitatively.
2. You can _always_ ask, “what is the expected recall probability of this fact”. Therefore you can _always_ sort a set of flashcards from most likely to be forgotten to least likely.
3. Being Bayesian lets us support a rich variety of quizzes fully analytically:
   - binary quizzes—pass/fail,
   - binomial quizzes—e.g., three points out of four,
   - even exotic noisy-binary quizzes that let you fully specify the odds of the student “passing” the quiz when they actually don’t know the answer (handy for deweighting multiple-choice vs. active recall, as well as for reader apps described above).

Note that Ebisu treats each flashcard’s memory as independent of the others. It can’t handle flashcard correlation or interference, alas, so you have to handle this in your application.

The hope is that Ebisu can be used by flashcard apps that continue to unleash the true potential of personalized learning and spaced reptition practice.

Now let’s jump into a more formal description of the mathematics, wihch will hopefully explain the Python [API](#api) presented above.

## Math

While much psychological literature has identified that forgetting follows power-law decay (e.g., probability of recall $t$ time units after last review $p(t) = (t+1)^{-α}$ for some positive shape parameter $α$), we start by discussing a simpler case, exponential decay, because we will eventually use an ensemble of exponentially-decaying functions to approximate a power law.

### Bernoulli quizzes

Let’s begin with a quiz. One way or another, we’ve picked a fact to quiz the student on, $t$ days after they last saw this fact. (The units of $t$ are arbitrary: it can be any positive real number.)

We’ll model the results of the quiz as a Bernoulli experiment—we’ll later expand this to a binomial experiment. So for Bernoulli quizzes, $x_t ∼ Bernoulli(p)$; $x_t$ can be either 1 (success) with probability $p_t$, or 0 (fail) with probability $1-p_t$. Let’s think about $p_t$ as the recall probability at time $t$. Then $x_t$ is a coin flip, with a $p_t$-weighted coin.

The [Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution) happens to be the [conjugate prior](https://en.wikipedia.org/wiki/Conjugate_prior) for the Bernoulli distribution. So if our _a priori_ belief about $p_t$ follow a Beta distribution, that is, if
$$p_t ∼ Beta(α, β)$$
for specific $α$ and $β$, then observing the quiz result updates our belief about the recall probability to be:
$$p_t | x_t ∼ Beta(α + x_t, β + 1 - x_t).$$

> **Aside 1** Notice that since $x_t$ is either 1 or 0, the updated parameters $(α + x_t, β + 1 - x_t)$ are $(α + 1, β)$ when the student correctly answered the quiz, and $(α, β + 1)$ when she answered incorrectly.
>
> **Aside 2** Even if you’re familiar with Bayesian statistics, if you’ve never worked with priors on probabilities, the meta-ness here might confuse you. What the above means is that, before we flipped our $p_t$-weighted coin (before we administered the quiz), we had a specific probability distribution representing the coin’s weighting $p_t$, _not_ just a scalar number. After we observed the result of the coin flip, we updated our belief about the coin’s weighting—it _still_ makes total sense to talk about the probability of something happening after it happens. Said another way, since we’re being Bayesian, something actually happening doesn’t preclude us from maintaining beliefs about what _could_ have happened.

This is totally ordinary, bread-and-butter Bayesian statistics. However, the major complication arises when the experiment took place not at time $t$ but $t_2$: we had a Beta prior on $p_t$ (probability of recall at time $t$) but the test is administered at some other time $t_2$.

How can we update our beliefs about the recall probability at time $t$ to another time $t_2$, either earlier or later than $t$?

### Moving Beta distributions through time

Our old friend Ebbinghaus comes to our rescue. According to the exponentially-decaying forgetting curve, the probability of recall at time $t$ is
$$p_t = 2^{-t/h},$$
for some notional half-life $h$. Let $t_2 = δ·t$. Then,
$$p_{t_2} = p_{δ t} = 2^{-δt/h} = (2^{-t/h})^δ = (p_t)^δ.$$
That is, to time-travel $p_t$ to time $t_2$, we raise it to the $δ = t_2 / t$ power.

Unfortunately, a Beta-distributed $p_t$ becomes _non_-Beta-distributed when raised to positive power $δ$. The plot below show what $p_{\text{1 week}} ∼ Beta(10, 10)$ looks like after

- two days (right histogram, $δ=0.3$, sharp peak around 0.8, very little density near 1.0),
- one week (middle histogram, $δ=1$, Bell-like curve centered at 0.5 and dying out by 0.2 and 0.8),
- and three weeks (left histogram, $δ=3$, peaking around 0.1 with a lot of support at 0, dying by 0.5).

![Three histograms per above](figures/pidelta.svg)

We could try to fit this non-Beta random variable with a Beta random variable to do the Bernoulli quiz update, but especially when over- or under-reviewing, the closest Beta fit is very poor. So let’s derive analytically the probability density function (PDF) for $p_t^δ$.

> Throughout this document, I use $P(x; X)$ to denote the density of the random variable $X$ as a function of the algebraic variable $x$.

Recall the conventional way to obtain the density of a [nonlinearly-transformed random variable](https://en.wikipedia.org/w/index.php?title=Random_variable&oldid=771423505#Functions_of_random_variables). Since the new random variable
$$p_{t_2} = g(p_t) = (p_t)^δ,$$
and the inverse of this transformation is
$$p_t = g^{-1}(p_{t_2}) = (p_{t_2})^{1/δ},$$
the transformed (exponentiated) random variable has probability density

$$
\begin{align*}
  P(p_{t_2}) &= P\left(g^{-1}(p_{t_2})\right) ⋅ \frac{∂}{∂p_{t_2}} g^{-1}(p_{t_2}) \\
             &= Beta(p_{t_2}^{1/δ}; α, β) ⋅ \frac{p_{t_2}^{1/δ - 1}}{δ},
\end{align*}
$$

since $P(p_t) = Beta(p_t; α, β)$, the Beta density on the recall probability at time $t$, and $\frac{∂}{∂p_{t_2}} g^{-1}(p_{t_2})^{1/δ} = \frac{p_{t_2}^{1/δ - 1}}{δ}$. Following some algebra, the final density is

$$
  P(p; p_t^δ) = \frac{p^{α/δ - 1} · (1-p^{1/δ})^{β-1}}{δ · B(α, β)},
$$

where $B(α, β) = Γ(α) · Γ(β) / Γ(α + β)$ is [beta function](https://en.wikipedia.org/wiki/Beta_function) (also the normalizing denominator in the Beta density—confusing, sorry), and $Γ(·)$ is the [gamma function](https://en.wikipedia.org/wiki/Gamma_function), a generalization of factorial.

[Robert Kern noticed](https://github.com/fasiha/ebisu/issues/5) that this is a [GB1](<https://en.wikipedia.org/w/index.php?title=Generalized_beta_distribution&oldid=889147668#Generalized_beta_of_first_kind_(GB1)>) (generalized Beta of the first kind) random variable:
$$p_t^δ ∼ GB1(p; 1/δ, 1, α; β)$$
When $δ=1$, that is, at exactly the half-life, recall probability is simply the initial Beta we started with.

We will use the density of $p_t^δ$ to reach our two most important goals:

- what’s the recall probability of a given fact right now?, and
- how do I update my estimate of that recall probability given quiz results?

> To check the above derivation in [Wolfram Alpha](https://www.wolframalpha.com), run `p^((a-1)/d) * (1 - p^(1/d))^(b-1) / Beta[a,b] * D[p^(1/d), y]`.
>
> To check it in [Sympy](https://live.sympy.org/), copy-paste the following into the Sympy Live Shell (or save it in a file and run):
>
> ```py
> from sympy import symbols, simplify, diff
> p_1, p_2, a, b, d, den = symbols('p_1 p_2 α β δ den', positive=True, real=True)
> prior_t = p_1**(a - 1) * (1 - p_1)**(b - 1) / den
> prior_t2 = simplify(prior_t.subs(p_1, p_2**(1 / d)) * diff(p_2**(1 / d), p_2))
> prior_t2  # or
> print(prior_t2)
> ```
>
> which produces `p_2**((α - δ)/δ)*(1 - p_2**(1/δ))**(β - 1)/(den*δ)`.
>
> And finally, we can use Monte Carlo to generate random draws from $p_t^δ$, for specific α, β, and δ, and comparing sample moments against the GB1's analytical moments per [Wikipedia](<https://en.wikipedia.org/w/index.php?title=Generalized_beta_distribution&oldid=889147668#Generalized_beta_of_first_kind_(GB1)>), $E\left[(p_{t}^{δ})^N\right]=\frac{B(α + δ N, β)}{B(α, β)}$:
>
> ```py
> (α, β, δ) = 5, 4, 3
> import numpy as np
> from scipy.stats import beta as betarv
> from scipy.special import beta as betafn
> prior_t = betarv.rvs(α, β, size=100_000)
> prior_t2 = prior_t**δ
> Ns = np.array([1, 2, 3, 4, 5])
> sampleMoments = [np.mean(prior_t2**N) for N in Ns]
> analyticalMoments = betafn(α + δ * Ns, β) / betafn(α, β)
> print(list(zip(sampleMoments, analyticalMoments)))
> ```
>
> which produces this tidy table of the first five non-central moments:
>
> | analytical | sample   | % difference |
> | ---------- | -------- | ------------ |
> | 0.2121     | 0.2122   | 0.042%       |
> | 0.06993    | 0.06991  | -0.02955%    |
> | 0.02941    | 0.02937  | -0.1427%     |
> | 0.01445    | 0.01442  | -0.2167%     |
> | 0.007905   | 0.007889 | -0.2082%     |
>
> We check both mathematical derivations and their programmatic implementations by comparing them against Monte Carlo as part of an extensive unit test suite in the code below.

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

## Deploy to PyPI

```sh
rm dist/* && python setup.py sdist bdist_wheel && python3 setup.py sdist bdist_wheel && twine upload dist/* --skip-existing
```
