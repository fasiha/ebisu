import ebisu


def test_init():
  init = ebisu.initModel(10, 10e3)
  assert ebisu.modelToPercentileDecay(init) > 10
