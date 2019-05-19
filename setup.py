from setuptools import setup

setup(name='ebisu',
      version='0.5.4',
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
      test_suite='nose.collector',
      tests_require=['nose'],
      install_requires=[
          'scipy', 'numpy'
      ],
      zip_safe=True)
