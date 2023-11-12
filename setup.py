# see https://blog.ionelmc.ro/2014/05/25/python-packaging/

from setuptools import setup, find_packages  # type:ignore
from glob import glob
from os.path import basename
from os.path import splitext

setup(
    name='ebisu',
    version='3.0.0rc1',
    description='Intelligent quiz scheduling',
    long_description=('Public-domain library for quiz scheduling with'
                      ' Bayesian statistics.'),
    keywords=('quiz review schedule spaced repetition system srs bayesian '
              'probability beta random variable'),
    url='http://github.com/fasiha/ebisu',
    author='Ahmed Fasih',
    author_email='wuzzyview@gmail.com',
    license='Unlicense',
    #
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    #
    test_suite='pytest',
    tests_require=['pytest'],
    install_requires=['scipy', 'numpy'],
    zip_safe=True)
