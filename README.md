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
    - [A single Beta-distributed atom](#a-single-beta-distributed-atom)
      - [Noiy-binary quizzes](#noiy-binary-quizzes)
    - [Power laws](#power-laws)
  - [Dev](#dev)
    - [Tests](#tests)
  - [Deploy to PyPI](#deploy-to-pypi)
  - [Bibliography](#bibliography)

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

> Ebisu is Bayesian, so it doesn't treat `firstHalflife` as "the truth", but rather as your probabilistic _pripr_ belief about the halflife, _prior_ to seeing any quizzes and considering only what you know about this student and this fact. More specifically, you are telling Ebisu that that after `firstHalflife` time units have elapsed, the student's memory will decay to a `Beta(α, β)` random variable, with both parameters, `α` and `β`, set to `initialAlphaBeta > 1`. Here's what that looks like visually for a few different `initialAlphaBeta`:
>
> ![Distribution of Beta random variables with parameters 1.25, 2, 5, and 10](./figures/betas.svg)
>
> This chart can be confusing if you're not used to thinking about probabilities on probabilities 😅! It shows that, no matter what you pick for `initialAlphaBeta`, the student's recall probability after one halflife is 50-50, since all four curves peak at 0.5. However, varying `initialAlphaBeta` encodes different beliefs about how _confident_ you are in your claim that this is the halflife. The default `initialAlphaBeta=2` gives you quite a lot of wiggle room: the student's probability of recall after one halflife might be as low as 20% or as high as 80%.
>
> In contrast, setting a high `initialAlphaBeta` says "I'm _really_ confident that the halflife I tell you is really the halflife", so for example, `initialAlphaBeta=10` has most its support between 40% and 60% recall probability. At the other end of the extreme, setting `initialAlphaBeta=1.25` (very close to the boundary of 1!) is _really_ uncertain about the student's probability of recall after one halflife without studying this fact. Most of its support is actually between 5% and 95%! The default of 2 is forgiving and gives you a lot of wiggle-room to be quite wrong about the true halflife (i.e., your prior as encoded by `firstHalflife` is allowed to be quite wrong).

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

So, a student has learned a few facts, and you've called `initModel` to create a model for each flashcard, and stored everything in a database.

Now, at this instant, what is the probability of recall for those flashcards?

```py
def predictRecall(
    prior: BetaEnsemble,
    elapsedTime: float,
    logDomain=True,
) -> float
```

Just give each model to the above function along with how long it's been since the student last saw this flashcard, `elapsedTime`; be sure to use the same units that you used for `initModel`!

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

There are two-ish distinct quiz types that Ebisu supports thanks to its Bayesian formulation, so this function has a lot of potential arguments. Whatever arguments you pass in, this function returns the new model: save it in your database, with the current timestamp, because future calls to `predictRecall` will need to know when it was last reviewed.

#### Simple binary quizzes

- On success, `updateRecall(model, 1, 1, elapsedTime)`.
- On failure, `updateRecall(model, 0, 1, elapsedTime)`.

That is, `successes` is 0 or 1 out of a `total=1` and `elapsedTime` in units consistent with `predictRecall` and `initModel`.

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

### A single Beta-distributed atom

We begin with a claim: after $t$ time units elapsed since their last review, a student's probability of recalling a fact is distributed according to $Beta(α, β)$ with $α = β$. More formally,
$$p_t ∼ Beta(α, β) \quad \text{with} \quad α = β$$
This is the same graph as above—what different Beta random variables look like for different $α=β$:

![Distribution of Beta random variables with parameters 1.25, 2, 5, and 10](./figures/betas.svg)

> We can call $t$ the "halflife" because after $t$ time units has elapsed, holding $α = β$ means the resultant random variable has mean 0.5.

At some _other_ time $t_2 ≠ t$, what is the probability of recall, $p_{t_2}$? Let's use our friend Ebbingaus' exponential decay. According to the exponentially-decaying forgetting curve, the probability of recall at time $t$ is
$$p_t = 2^{-t/h},$$
for some notional half-life $h$. Let $t_2 = δ·t$. Then,
$$p_{t_2} = p_{δ t} = 2^{-δt/h} = (2^{-t/h})^δ = (p_t)^δ.$$
That is, to time-travel $p_t$ to time $t_2$, we raise it to the $δ = t_2 / t$ power.

Working through the calculus-heavy technique of obtainining the density of a [nonlinearly-transformed random variable](https://en.wikipedia.org/w/index.php?title=Random_variable&oldid=771423505#Functions_of_random_variables), we can show that if $P(p_t) = Beta(p_t; α, β)$, i.e., the probability density function of $p_t$ is the Beta density, then
$$P(p; p_t^δ) = \frac{p^{α/δ - 1} · (1-p^{1/δ})^{β-1}}{δ · B(α, β)},$$
where $B(α, β) = Γ(α) · Γ(β) / Γ(α + β)$ is [Beta function](https://en.wikipedia.org/wiki/Beta_function), not to be confused with the Beta distribution (sorry), and $Γ(·)$ is the [gamma function](https://en.wikipedia.org/wiki/Gamma_function), a generalization of factorial.

[Robert Kern noticed](https://github.com/fasiha/ebisu/issues/5) that this is a [GB1](<https://en.wikipedia.org/w/index.php?title=Generalized_beta_distribution&oldid=889147668#Generalized_beta_of_first_kind_(GB1)>) (generalized Beta of the first kind) random variable:
$$p_t^δ ∼ GB1(p; 1/δ, 1, α; β)$$
When $δ=1$, that is, at exactly the half-life, recall probability is simply the initial Beta we started with.

If the recall probability after $t$ time units $p_t$ is distributed according to $Beta(α, β)$, then the expected value (the mean) of $p_{t_2} = p_t^δ$ <a name="precall-formula">is</a>
$$E[p_t^δ] = \frac{B(α+δ, β)}{B(α,β)} = \frac{Γ(α + β)}{Γ(α)}  \frac{Γ(α + δ)}{Γ(α + β + δ)}.$$
In other words, this is the expected recall probability at any time $t_2$, given that we believe the recall at time $t$ to follow $Beta(α, β)$.

Now, at time $t_2$, we have a quiz. There are two types of quizzes we want to model, but lets start with a binomial quiz, parameterized by $k$ successes out of $n$ attempts, with $0 ≤ k ≤ n$ and $n ≥ 1$, both integers. This is equivalent to flipping the $p_{t_2}$-weighted coin a total of $n$ times and winning $k$ tosses.

By application of Bayes rule, the posterior is
$$Posterior(p|k, n) = \frac{Prior(p) · Lik(k|p,n)}{\int_0^1 Prior(p) · Lik(k|p,n) \, dp}.$$
Here,

- “prior” refers to the GB1 density $P(p_t^δ)$ derived above.
- $Lik$ is the binomial likelihood: $Lik(k|p,n) = \binom{n}{k} p^k (1-p)^{n-k}$. This is what you see on Wikipedia etc., the PMF (probability mass function) of a binomial random variable.
- The denominator is the marginal probability of the observation $k$.

Combining all these into one expression, we have:

$$
Posterior(p|k, n) = \frac{
    p^{α/δ - 1} (1-p^{1/δ})^{β - 1} p^k (1-p)^{n-k}
  }{
    \int_0^1 p^{α/δ - 1} (1-p^{1/δ})^{β - 1} p^k (1-p)^{n-k} \, dp
  },
$$

where note that the big integrand in the denominator is just the numerator.

We use two helpful facts now. The more important one is that
$$\int_0^1 p^{α/δ - 1} (1-p^{1/δ})^{β - 1} \, dp = δ ⋅ B(α, β),$$
when $α, β, δ > 0$. We’ll use this fact several times in what follows—you can see the form of this integrand in the big integrand in the above posterior.

The second helpful fact gets us around that pesky $(1-p)^{n-k}$. By applying the [binomial theorem](https://en.wikipedia.org/w/index.php?title=Binomial_theorem&oldid=944317290#Theorem_statement), we can see that
$$\int_0^1 f(x) (1-x)^n \, dx = \sum_{i=0}^{n} \left[ \binom{n}{i} (-1)^i \int_0^1 x^i f(x) \, dx \right],$$
for integer $n > 0$.

Putting these two facts to use, we can show that the posterior at time $t_2$ is

$$
  Posterior(p; p_{t_2}|k, n) = \frac{
    \sum_{i=0}^{n-k} \binom{n-k}{i} (-1)^i p^{α / δ + k + i - 1} (1-p^{1/δ})^{β - 1}
  }{
    δ \sum_{i=0}^{n-k} \binom{n-k}{i} (-1)^i ⋅ B(α + δ (k + i), \, β)
  }.
$$

Given a Beta-prior on the recall probability after $t$ time units, and a binomial quiz at some other elapsed time $t_2$, the above gives the updated posterior on recall probability after $t_2$ time units.

We could work with this but let's aim for one more enhancement.

I'd prefer to have the posterior recall probability after some arbitrary time $t'$ has elapsed since last review, totally independent of both $t$ (the time of the original prior) as well as $t_2$ (the time of the quiz). We can follow the rules of [nonlinear transforms of random variables](https://en.wikipedia.org/w/index.php?title=Random_variable&oldid=771423505#Functions_of_random_variables) to get the posterior at this new $t'$, and just as we have $δ=t_2/t$ to go from $t$ (the original prior's time horizon) to $t_2$ (the quiz time), let $ε=t' / t_2$ to go from $t_2$ (the quiz time) to $t'$ (some arbitrary time for the posterior):

$$
\begin{align*}
  P(p; p_{t'} | k_{t_2}, n_{t_2})
  &=
  Posterior \left(p^{1/ε}; p_{t_2}|k_{t_2}, n_{t_2} \right) ⋅ \frac{1}{ε} p^{1/ε - 1}
  \\
  &= \frac{
    \sum_{i=0}^{n-k} \binom{n-k}{i} (-1)^i p^{\frac{α + δ (k + i)}{δ ε} - 1} (1-p^{1/(δε)})^{β - 1}
  }{
    δε \sum_{i=0}^{n-k} \binom{n-k}{i} (-1)^i ⋅ B(α + δ (k + i), \, β)
  }.
\end{align*}
$$

The denominator is the same in this $t'$-time-shifted posterior since it’s just a normalizing constant (and not a function of probability $p$) but the numerator retains the same shape as the original, allowing us to use one of our helpful facts above to derive this transformed posterior’s moments. The $N$th moment, $E[p_{t'}^N]$, is:

$$
  m_N = \frac{
    \sum_{i=0}^{n-k} \binom{n-k}{i} (-1)^i ⋅ B(α + (i+k)δ + N δ ε, \, β)
  }{
    \sum_{i=0}^{n-k} \binom{n-k}{i} (-1)^i ⋅ B(α + (i+k)δ, \, β)
  }.
$$

With these moments of our final posterior at arbitrary time $t'$ in hand, we can moment-match to recover a Beta-distributed random variable that serves as the new prior. Recall that a distribution with mean $μ$ and variance $σ^2$ can be fit to a Beta distribution with parameters:

- $\hat α = (μ(1-μ)/σ^2 - 1) ⋅ μ$ and
- $\hat β = (μ(1-μ)/σ^2 - 1) ⋅ (1-μ)$.

To summarize the update step: you started with a flashcard whose memory model was $[α, β, t]$, that is, the prior on recall probability after $t$ time units since the previous encounter is $Beta(α, β)$. At time $t_2$, you administer a quiz session that results in $k$ successful recollections of this flashcard, out of a total of $n$. Therefore, the updated model is, for any arbitrary time $t'$,

$$[μ (μ(1-μ)/σ^2 - 1), \, (1-μ) (μ(1-μ)/σ^2 - 1), \, t']$$

and for

- $δ = t_2/t$,
- $ε=t'/t_2$, where both
- $μ = m_1$ and
- $σ^2 = m_2 - μ^2$ come from evaluating the appropriate $m_N$:
- $m_N = \frac{
\sum_{i=0}^{n-k} \binom{n-k}{i} (-1)^i ⋅ B(α + (i+k)δ + N δ ε, \, β)
}{
\sum_{i=0}^{n-k} \binom{n-k}{i} (-1)^i ⋅ B(α + (i+k)δ, \, β)
}$.

In practice, we’ll choose $t'$ to be the posterior’s halflife, i.e., we’ll pick $t'$ such that the new $\hat α ≈ \hat β.$

Whew, that was a whirlwind tour through some mathematical thickets. Let's take a look at what this looks like in practice. Suppose we start with an initial model, that the student’s memory has halflife of one day, that is, a day after studying this fact, we expect the student's recall probability to follow $Beta(3, 3)$. The following plot shows the posterior halflife for a range of quiz times, ranging from a half hour to four days, for

- the binary quiz case, i.e., pass and fail, as well as
- the binomial $n=2$ case, i.e., the student gets $k$ = 0, 1, or 2 points.

![Comparing the posterior halflife after a binary pass/fail and a binomial 0, 1, and 2 points out of 2](./figures/binaryBinomial.svg)

In the binary case, if the student passes the quiz a few minutes after last studying the fact, Ebisu is not impressed and the posterior halflife barely rises. Similarly, if the student _fails_ a quiz way after four days of not studying (that is, far beyond the fact's halflife), Ebisu is also not surprised at this failure: the posterior halflife drops a bit, not a lot.

Ebisu is surprised in the opposite cases: if the student fails the quiz a few minutes after studying, the posterior halflife drops a lot, and similarly if the student passes the quiz after several days of not studying it, the halflife rises a lot.

The _binomial_ case evinces the exact same behaviors, but more exaggerated. Notice that the 0 out of 2 points case produces a posterior halflife curve that's strictly lower than the 0 out of 1 points (binary failure) case, while the 2 out of 2 points curve is strictly higher than the 1 out of 1 points (binary pass) curve. The 1 out of 2 points curve is kind of half-way between the binary curves. Getting 1 out of 2 points on a quiz around one halflife (one day) after studying leaves the posterior halflife unmoved.

#### Noiy-binary quizzes

Can we imagine a quiz type where the student could score 0 or 1 point (out of max 1) but also somewhere in between? As luck would have it, Stack Exchange user [@mef](https://stats.stackexchange.com/a/419320) has invented a lovely way to model this and it is quite useful in capturing some advanced quiz types.

Let $x ∼ \mathrm{Bernoulli}(p_t)$ be a “true” Bernoulli draw representing the answer to the question, “does the student _really_ know this fact?” ($p_t$ here is the time-traveled Beta random variable, i.e., a GB1 random variable.)

But this random variable is hidden from us: we don’t observe $x$. Rather, we observe a quiz, a “noisy report”, $z|x ∼ \mathrm{Bernoulli}(q_x)$ where

- $q_1 = P(z = 1 | x = 1)$, that is, the probability of us observing a successful quiz when the student actually _does_ knows the fact, while
- $q_0 = P(z = 1 | x = 0)$, i.e., the probability of us observing a successful quiz when the student has in fact _forgotten_ the fact.

In signal processing terms, the true but hidden result $x$ goes through a noisy channel, which might flip the bit, and what we observe is the output of that channel, $z$.

In the plain binary case without fuzziness, $q_1 = 1$ and $q_0 = 0$, but in the soft-binary case, these two parameters are independent and free for you to specify as any numbers between 0 and 1 inclusive.

Given the probability of recall at the time of the quiz $t_2$ is $p_{t_2}∼GB1$ and with known $α$, $β$, $q_1$, and $q_0$, we can ask what the posterior $p_{t_2} | z$ is. According to Bayes,

$$
P(h|z) = \frac{Lik(z|p_{t_2})Prior(p_{t_2})}{∫_0^∞ Lik(z|p_{t_2})Prior(p_{t_2}) \,\mathrm{d}h},
$$

with the "prior" being the GB1 density and the likelihood given by

$$
\begin{split}
Lik(z|p_{t_2}) &= P(z|x) ⋅ P(x|p_{t_2})
  \\
  &= \mathrm{Bernoulli}(z; q_x) ⋅ \mathrm{Bernoulli}\left(x; p_{t_2}\right).
\end{split}
$$

Let’s break this likelihood into its two cases: first, for observed _failed_ quizzes,

$$
\begin{align*}
  Lik(z=0 | p) &= P(z=0|x=0) P(x=0|p) + P(z=0|x=1) P(x=1|p) \\
               &= (1-q_0)(1-p) + (1-q_1) p.
\end{align*}
$$

And following the same pattern, for observed _successes_:

$$
\begin{align*}
  Lik(z=1| p) &= P(z=1|x=0) P(x=0|p) + P(z=1|x=1) P(x=1|p) \\
                 &= q_0 (1-p) + q_1 p.
\end{align*}
$$

As in the binomial case, we want the flexibility to time-travel it to any time $t' = ε ⋅ t_2$. We’ve done this twice already—first to transform the Beta prior on recall after $t$ to $t_2 = δ ⋅ t$, and then again to transform the _binomial_ posterior from the quiz time $t_2$ to any $t' = ε ⋅ t_2$. Let’s do it a third time. The pattern is the same as before:

$$
  P(p; p_{t'}|z_{t_2}) ∝ Prior(p^{1/ε}) ⋅ Lik(p^{1/ε}) ⋅ \frac{1}{ε} p^{1/ε - 1}
$$

where the $∝$ symbol is read “proportional to” and just means that the expression on the right has to be normalized (divide it by its integral) to ensure the result is a true probability density whose definite integral sums to one.

We can represent the likelihood of any $n=1$ quiz—binary and noisy!—as $Lik(z|p) = r p + s$ for some $r$ and $s$. Then,

$$
  P(p; p_{t'}|z_{t_2}) = \frac{
    \left( r p^{\frac{α + δ}{δ ε} - 1} + s p^{\frac{α}{δ ε}-1} \right)
    \left( 1-p^{\frac{1}{δ ε}} \right)^{β - 1}
  }{
    δ ε (r B(α + δ, β) + s B(α, β))
  }.
$$

The normalizing denominator comes from $\int_0^1 p^{a/x - 1} (1-p^{1/x})^{b - 1} dp = x ⋅ B(a, b)$, which we also used in the binomial case above. This fact is also very helpful to evaluate the moments of this posterior:

$$
  m_N = E\left[ p_{t'}^N\right] = \frac{
    c B(α + δ(1 + N ε), β) + s B(α + δ N ε, β)
  }{
    r B(α + δ, β) + s B(α, β)
  },
$$

with

- for $z=0$, i.e., a failed quiz,
  - $r = q_0 - q_1$ (-1 for a binary non-fuzzy quiz)
  - $s = 1-q_0$ (1 for a binary non-fuzzy quiz).
- For $z=1$, a successful quiz,
  - $r = q_1 - q_0$ (1 for a binary non-fuzzy quiz)
  - $s = q_0$ (0 for a binary non-fuzzy quiz).

With these moments, we can once again match the first and second moments to get our new model, $[\hat α, \hat β, t']$:

- $δ = t_2/t$, i.e., time of quiz divided by the original model time,
- $ε=t'/t_2$, i.e., the time you want the posterior at divided by the time of quiz,
- $\hat α = (μ(1-μ)/σ^2 - 1) ⋅ μ$ and
- $\hat β = (μ(1-μ)/σ^2 - 1) ⋅ (1-μ)$, where
- $μ = m_1$
- and $σ^2 = m_2 - m_1^2$.

We again note that both $q_1 = P(z = 1 | x = 1)$ and $q_0 = P(z = 1 | x = 0)$ are _free_ parameters, and apps have total flexibility in specifying these. In Ebisu’s API presented above, both $z$ and $q_1$ are encoded without loss of generality in `0 <= successes <= 1`:

- $z=1$ if `successes > 0.5`, otherwise $z=0$.
- $q_1$ is `max(successes, 1 - successes)`.

Therefore if `successes = 0.1`, then we know $z=0$ and $q_1 = 0.9$.

Meanwhile, $q_0$ is provided in a keyword argument and for the sake of developer experience, $q_0=1-q_1$ is picked as a default when none is provided. While this default is ad hoc, it does have the nice property that `successes` between 0 and 1 will smoothly and symmetrically (around 0.5) scale the posterior halflife between the binary fail/pass cases. Also, as a satisfying bonus, a _totally_ uninformative quiz with `successes = 0.5` results in _no change_ to the prior, i.e., $α' = α$ and $β' = β$!

The above is a very fast tour through the mathematics of taking a single Beta random variable for a particular time through two distinct kinds of quizzes: binomial quizzes and noisy-binary quizzes. This is what Ebisu v2 and previous historically were. However, we need more.

### Power laws

Everything above has relied on the _exponential_ forgetting model, where our beliefs about the recall probability at time $t$ are converted to beliefs on recall probabilty at other times via $p_{δ \cdot t} = p_t^δ$, even though considerable research has shown forgetting is a power-law phenomenon. In this section, we will show how a staggered sequence of exponentials leads us to the power law we’d like to govern Ebisu recall.

To set the stage first—recall that there is a truly _fundamental_ difference between _exponential_ decay, where the time factor $t$ is in the exponent, versus power-law decay, where $t$ is in the base. An exponential like $2^{-t}$ decays _incredibly_ quickly—you will recall what Einstein apparently said about exponentials being the most powerful force in the universe. After seven halflives, the probability of recall has dropped less than 1%: $2^{-7} = 1/128$. Meanwhile, power laws decay much more slowly: $(t+1)^{-1}$ has the same halflife as $2^{-t}$ (both have decayed to 0.5 at $t=1$) but after seven halflives, the probability of recall for the power law is still $1/8$, i.e., 12.5%, more than an order of magnitude higher than 0.8%!

> This captures the experience we’ve all had where lots of things just “stick in your head”, i.e., have very durable memories, despite having not studied them intensively.

Mozer et al. in both their 2009 NIPS conference paper and their 2014 <cite>Psychological Science</cite> journal article (see [bibliography](#bibliography)) propose a model of recall that uses a _series_ of exponentials, which they call a “cascade of leaky integrators” to invoke previous psychology research. We will adopt this by constructing a model consisting on not just one Beta random variable but a weighted ensemble of them over a wide range of halflives.

Ebisu constructs models with $N$ distinct $Beta(α, β)$ random variables governing the expected recall probability over $N$ logarithmically-spaced time horizons, each with a logarithmically-decreasing weight. For example, an ensemble with five atoms, with initial halflives ranging from 10 hours to 100,000 hours (11 years), and weights ranging from 50% for the shortest-duration atom to 4% for the longest-duration one, generates the following five curves for recall probability, with the ensemble's overall recall probability being just the weighted sum of all atoms':

![Between 0 and 200 hours since last review, the recall probability for five atoms and the overall ensemble](./figures/power-linear.svg)

The largest-weighted atom is also the fastest-decaying, whereas the lowest-weighted atom decays very slowly. The curves are exponential-ish because they describe the $E[p_t^δ]$ curve we computed above for $p_t∼Beta(α, β)$. The first 10-hour-halflife atom's recall probabily quickly falls below each less-weighted atom's till after 75 hours it's dropped below the 11-year-halflife atom' recall probability. This results in the overall ensemble's recall probability decaying quite quickly initially but have a _long tail_, propped up by the lower-weighted long-duration atoms.

Switching the above plot’s x and y scales to log-log gives and zooming out to see more time gives us this—

![The same plot above but with the x axis going from 0.1 hours to a million hours. In the log-log domain, each exponential starts out flat (full recall), then quickly dies away. The overall ensemble curve though starts out flat at 100% (full recall), has a long linear section (power law), before it drops off steeply with the last exponential](./figures/power-log.svg)

Recall that in a log-log plot, a true power law is a straight line, and the ensemble's recall probability describes a nice, sloping line for a huge chunk of time, from roughly 10 hours to roughly 10 years. For very short intervals between study, we have full recall (flat), and for intervals longer than our longest-duration atom, we have a rapidly-decaying exponential, and indeed any natural power law will have to taper off somewhere.

It's through the logarithmic spacing of atoms' initial halflives and weights that, despite each atom describing an exponentially-decaying memory, the overall ensemble achieves this smooth power law recall that still assigns a one-ish percent probability of remembering this fact a decade after last studying it.

When we discussed a single Beta random variable governing recall probability at some time, we parameterized it with three numbers: $[α, β, t]$; in words, the recall probability $t$ time after last study was $p_t ∼ Beta(α, β)$. To parameterize an ensemble of $N$ weighted Beta-distributed random variables, we have $[α_i, β_i, t_i, w_i]$ for $i=1, 2, \cdots, N$.

An important feature of weighted mixtures is the weights must sum to 1. Ebisu's `initModel` API requests a `firstWeight`, i.e., $0 < w_1 < 1$, and computes other weights by finding $x$ that achieves $\min_x \left| 1-∑_{i=0}^{N-1} w_1 x^{i} \right|$. As far as I know this can’t be done analytically so we do a quick grid search.

The ensemble's overall recall probability at time $t_2$ is simply the weighted sum of individual atoms’ recall probability:
$$E[p_{t_2}] = ∑_{i=1}^N w_i E\left[p_i^{δ_i}\right]$$
where we denote the $i$th atom’s probability at time $t_2$ as $p_i^{δ_i}$ for $δ_i = t_2 / t_i$, and $p_i ∼ Beta(α_i, β_i)$. The expectation in the sum is give [above](#precall-formula).

The final question remains: how do we _update_ an ensemble after a binomial or noisy-binary quiz?

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

## Bibliography

While most citations are given inline above, this section contains academic papers, to whose PDFs I want to provide multiple links.

Lindsey, R. V., Shroyer, J. D., Pashler, H., & Mozer, M. C. (2014). Improving Students’ Long-Term Knowledge Retention Through Personalized Review. <cite>Psychological Science</cite>, 25(3), 639–647. [DOI](https://doi.org/10.1177/0956797613504302), [academic copy](https://home.cs.colorado.edu/~mozer/Research/Selected%20Publications/reprints/LindseyShroyerPashlerMozer2014Published.pdf). The authors also share some very interesting mathematical details as “Additional Methods” under [Supplemental Material](https://journals.sagepub.com/doi/10.1177/0956797613504302#supplementary-materials) on SagePub.

Michael C. Mozer, Harold Pashler, Nicholas Cepeda, Robert Lindsey, and Ed Vul. 2009. Predicting the optimal spacing of study: a multiscale context model of memory. In <cite>Proceedings of the 22nd International Conference on Neural Information Processing Systems (NIPS'09)</cite>. Curran Associates Inc., Red Hook, NY, USA, 1321–1329. [DOI](https://dl.acm.org/doi/10.5555/2984093.2984242), [academic copy](https://home.cs.colorado.edu/~mozer/Research/Selected%20Publications/reprints/MozerPashlerCepedaLindseyVul2009.pdf).
