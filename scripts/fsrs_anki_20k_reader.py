import os
import csv
from collections import defaultdict, namedtuple
import random
from typing import DefaultDict
from natsort import natsorted


def custom_walk(directory_path: str):
  assert os.path.exists(directory_path)
  for root, dirs, files in os.walk(directory_path):
    dirs[:] = natsorted(dirs)  # Natural sort for directories
    files = natsorted(files)  # Natural sort for files
    for file in files:
      if file.endswith('.csv'):
        yield os.path.join(root, file)


# Define a named tuple for the card data
CardData = namedtuple('CardData',
                      ['card_id', 'review_th', 'delta_t', 'rating', 'state', 'duration'])


def allCards(directory_path: str, train_percent: float = 1.0, seed=None):
  assert 0 < train_percent <= 1

  rng = random.Random(seed)
  # Walk through the directory recursively
  for file in custom_walk(directory_path):
    with open(file, newline='') as csvfile:
      reader = csv.DictReader(csvfile)
      # Use a defaultdict to collect rows by card_id
      card_groups: DefaultDict[int, list[CardData]] = defaultdict(list)
      for row in reader:
        # Convert the row into a named tuple or dict
        card = CardData(
            card_id=int(row['card_id']),
            review_th=int(row['review_th']),
            delta_t=int(row['delta_t']),
            rating=int(row['rating']),
            state=int(row['state']),
            duration=int(row['duration']))
        card_groups[card.card_id].append(card)

      # Yield each group of rows by card_id
      for card_id, cards in card_groups.items():
        if train_percent == 1 or rng.random() <= train_percent:
          yield cards


if __name__ == "__main__":
  it = allCards(os.path.join(os.getenv('FSRS_PATH', '.'), 'dataset'))
  print(next(it))
  print(next(it))
