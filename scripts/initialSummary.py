"""
python -m pip install humanize "ebisu>=3rc"
"""

import datetime as dt
import humanize
import ebisu

for a in ebisu.initModel(10):
  print(
      f'- halflife: {humanize.precisedelta(dt.timedelta(hours=a.time))}; weight {2**a.log2weight*100:.2g}%'
  )
