from setuptools import setup

setup(name='ebisu',
      version='2.1.0',
      description='Intelligent quiz scheduling',
      long_description=('Public-domain library for quiz scheduling with'
                        ' Bayesian statistics.'),
      keywords=('quiz review schedule spaced repetition system srs bayesian '
                'probability beta random variable'),
      url='http://github.com/fasiha/ebisu',
      author='Ahmed Fasih',
      author_email='wuzzyview@gmail.com',
      license='Unlicense',
      packages=['ebisu'],
      test_suite='unittest',
      install_requires=[
          'scipy', 'numpy', 'dataclasses-json'
      ],
      extras_require={'stan': ['cmdstanpy']},
      zip_safe=True)
