import os
import csv
from collections import defaultdict, namedtuple
import random
from typing import Callable, DefaultDict, Iterable, TypeVar
from natsort import natsorted
from itertools import takewhile
# from functools import reduce


def custom_walk(directory_path: str):
  assert os.path.exists(directory_path)
  for root, dirs, files in os.walk(directory_path):
    dirs[:] = natsorted(dirs)  # Natural sort for directories
    files = natsorted(files)  # Natural sort for files
    for file in files:
      if file.endswith('.csv'):
        yield os.path.join(root, file)


# Define a named tuple for the card data
CardData = namedtuple(
    'CardData',
    ['card_id', 'review_th', 'delta_t', 'rating', 'state', 'duration', "delta_t_sec", "file"])


def loadFile(file: str) -> DefaultDict[int, list[CardData]]:
  with open(file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    # Use a defaultdict to collect rows by card_id
    card_groups: DefaultDict[int, list[CardData]] = defaultdict(list)

    for row in reader:
      card = CardData(
          card_id=int(row['card_id']),
          review_th=int(row['review_th']),
          delta_t=int(row['delta_t']),
          rating=int(row['rating']),
          state=int(row['state']),
          duration=int(row['duration']),
          delta_t_sec=int(row['delta_t_sec']),
          file=file,
      )
      card_groups[card.card_id].append(card)
  return card_groups


def allCards(
    directory_path: str,
    #  min_delta_t_sec=-1,
    min_reviews=1,
    card_fraction: float = 1.0,
    user_fraction=1.0,
    seed=None,
    merge_trailing_successes=True):
  """Generator yielding a list of cards

  `directory_path` is the path to your clone of `FSRS-Anki-20k/` (just needs the
  `dataset/` subdirectory so you can delete the rest of the contents if you're low on disk
  space)

  `min_delta_t_sec`: only yield reviews with `delta_t >= min_delta_t_sec`. (Reminder:
  `delta_t_sec=-1` means "learned".)

  `min_reviews`: don't yield cards with fewer than these number of reviews (reminder: the
  first will be the initial learn step). This applies after the `min_delta_t_sec` filter.

  `user_fraction` and `card_fraction` (both between 0 and 1): yield only this percent of
  users and cards respectively. `card_fraction` is applied *after* `min_delta_t_sec` and
  `min_reviews` filters.

  `seed`: initialization for user/card sampler. Use for repeatable runs.
  """
  assert 0 < card_fraction <= 1
  assert 0 < user_fraction <= 1
  assert 0 < min_reviews
  # assert -1 <= min_delta_t_sec

  rng = random.Random(seed)
  # Walk through the directory recursively
  for file in custom_walk(directory_path):
    include_user = user_fraction == 1 or rng.random() <= user_fraction
    if not include_user:
      continue

    card_groups = loadFile(file)
    for card_id, cards in card_groups.items():
      combined = combine(cards, merge_trailing_successes=merge_trailing_successes)[1:]
      include_card = (
          len(combined) >= min_reviews and (card_fraction == 1 or rng.random() <= card_fraction))
      if not include_card:
        continue
      yield combined


T = TypeVar('T')


def split_by(split_pred: Callable[[T, list[T]], bool], lst: Iterable[T]) -> list[list[T]]:
  "Allows each element to decide if it wants to not be in previous partition"
  lst = iter(lst)
  try:
    x = next(lst)
  except StopIteration:  # empty iterable (list, zip, etc.)
    return []
  ret: list[list[T]] = [[x]]
  for x in lst:
    if split_pred(x, ret[-1]):
      ret.append([x])
    else:
      ret[-1].append(x)
  return ret


def find_first(pred: Callable[[T], bool], l: list[T]) -> int:
  for i, x in enumerate(l):
    if pred(x):
      return i
  return -1


def find_last(pred: Callable[[T], bool], l: list[T]) -> int:
  for i in range(len(l) - 1, -1, -1):
    if pred(l[i]):
      return i
  return -1


def combine(quizzes: list[CardData],
            window_in_seconds=4 * 3600,
            merge_trailing_successes=True) -> list[CardData]:
  assert quizzes[0].delta_t_sec == -1

  absolute_seconds: list[float] = []
  for q in quizzes:
    if q.delta_t_sec == -1:
      absolute_seconds.append(0)
    else:
      absolute_seconds.append(absolute_seconds[-1] + q.delta_t_sec)

  def splitter(quizTime: tuple[CardData, float], qs: list[tuple[CardData, float]]) -> bool:
    t = quizTime[1]
    prev = qs[-1]
    prev_time = prev[1]
    return (t - prev_time) > window_in_seconds

  quizTimes: list[tuple[CardData, float]] = []
  for group in split_by(splitter, zip(quizzes, absolute_seconds)):
    first_fail_idx = find_first(lambda x: x[0].rating == 1, group)
    last_fail_idx = find_last(lambda x: x[0].rating == 1, group)

    if first_fail_idx >= 0:
      assert 0 <= first_fail_idx <= last_fail_idx
      assert last_fail_idx < len(group)
      failure_qt = group[first_fail_idx]
      # append first fail
      quizTimes.append(failure_qt)
      if merge_trailing_successes:
        # append all passes after last fail (which might or might not be first fail)
        quizTimes.extend(group[(last_fail_idx + 1):])
    else:
      quizTimes.extend(group)

  ret: list[CardData] = []
  for i, (q, t) in enumerate(quizTimes):
    if i == 0:
      ret.append(q)
    else:
      prev_qt = quizTimes[i - 1]
      delta_sec = t - prev_qt[1]
      newq = q._replace(delta_t_sec=delta_sec, delta_t=int(round(delta_sec / (24 * 3600))))
      ret.append(newq)
  return ret


if __name__ == "__main__":
  it = allCards(os.path.join(os.getenv('FSRS_PATH', '.'), 'dataset'))
  # print(next(it))
  # print(next(it))
  cards = loadFile(os.path.join(os.getenv('FSRS_PATH', '.'), 'dataset', '1', '35.csv'))
  card = cards[252]
  print(card)
  print(combine(card))
