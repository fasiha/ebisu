from copy import deepcopy
import pylab as plt
import numpy as np
from typing import Any
from pprint import pprint

import ebisu
import ebisu3wmax
import ebisu2beta
import ebisu3boost
import utils

plt.ion()
# np.seterr(all='raise')
np.set_printoptions(precision=4, suppress=True)


def vizRank(d: dict[str, list[int]]):
  keyToId: None | dict[int, int] = None
  ret = dict()
  for key in d:
    if keyToId is None:
      keyToId = {k: i for i, k in enumerate(d[key])}
    ret[key] = [keyToId[k] for k in d[key]]
  return ret


if __name__ == "__main__":
  df = utils.sqliteToDf('collection.anki2', True)
  print(f'loaded SQL data, {len(df)} rows')

  train, TEST_TRAIN = utils.traintest(df)
  print(f'split flashcards into train/test, {len(train)} cards in train set')

  models: dict[str, dict[int, Any]] = {
      'v3wmax': dict(),
      'v2beta': dict(),
      'v3boost': dict(),
      'v3betas': dict(),
      'v3gammas': dict(),
  }

  # Unix millis, elapsed hours, key, Anki result (1-4) or "first learned" (-1)
  history: list[tuple[float, float, int, int]] = []

  for card in train[:20]:
    history.append(((card.absts_hours[0] - card.dts_hours[0]) * 3600e3, 0, card.key, -1))
    history.extend((t * 3600e3, dt, card.key, x)
                   for t, dt, x in zip(card.absts_hours, card.dts_hours, card.results))
  history.sort(key=lambda tup: tup[0])

  n = 0
  ranked: dict[str, list[tuple[int, float]]] = dict()
  lastSeenMillis: dict[int, float] = dict()
  for (unixMillis, elapsedHours, key, ankiResult) in history:
    if ankiResult < 0:
      # first learned
      assert key not in models['v3wmax']
      models['v3wmax'][key] = ebisu3wmax.initModel(initHlMean=12., now=unixMillis)
      models['v2beta'][key] = ebisu2beta.defaultModel(12., 2.0)
      models['v3boost'][key] = ebisu3boost.initModel(12.0, 3.0, now=unixMillis)
      models['v3betas'][key] = ebisu3wmax.initModel(0.02, now=unixMillis)
      models['v3gammas'][key] = ebisu.initModel(halflife=10, now=unixMillis)
    else:
      n += 1
      ranked = dict()

      ranked['v3wmax'] = [(k, ebisu3wmax.predictRecall(model, now=unixMillis))
                          for k, model in models['v3wmax'].items()]
      ranked['v2beta'] = [(k,
                           ebisu2beta.predictRecall(model,
                                                    (unixMillis - lastSeenMillis[k]) / 3600e3))
                          for k, model in models['v2beta'].items()]
      ranked['v3boost'] = [(k, ebisu3boost.predictRecall(model, now=unixMillis))
                           for k, model in models['v3boost'].items()]
      ranked['v3betas'] = [(k, ebisu3wmax.predictRecallBetas(model, now=unixMillis))
                           for k, model in models['v3betas'].items()]
      ranked['v3gammas'] = [(k, ebisu.predictRecall(model, now=unixMillis))
                            for k, model in models['v3gammas'].items()]

      if len(ranked['v3wmax']) >= 3:
        for algo in ranked:
          ranked[algo].sort(key=lambda v: v[1])
        print(f'{unixMillis=}')
        pprint(ranked)
        viz = {
            k: " ".join(map(str, v))
            for k, v in vizRank({k: [x[0] for x in v] for k, v in ranked.items()}).items()
        }
        pprint(viz)

      oldModels = deepcopy(models)

      successes = 1 if ankiResult >= 2 else 0
      models['v3wmax'][key] = ebisu3wmax.updateRecall(
          models['v3wmax'][key], successes, now=unixMillis)
      models['v2beta'][key] = ebisu2beta.updateRecall(
          models['v2beta'][key], successes, total=1, tnow=elapsedHours)
      models['v3boost'][key] = ebisu3boost.updateRecall(
          models['v3boost'][key], successes, now=unixMillis)
      models['v3betas'][key] = ebisu3wmax.updateRecallBetas(
          models['v3betas'][key], successes, now=unixMillis)
      models['v3gammas'][key] = ebisu.updateRecall(
          models['v3gammas'][key], successes, now=unixMillis)

    lastSeenMillis[key] = unixMillis

  cardKey = lambda v: v[0]

  # plt.figure()
  # plt.scatter([v[1] for v in sorted(ranked["v3pred"], key=cardKey)],
  #             [v[1] for v in sorted(ranked["v3"], key=cardKey)])
