import os
from setuptools import setup


def read(fname):
    """
    Helper to read README
    """
    return open(os.path.join(os.path.dirname(__file__), fname)).read().strip()


setup(
    name='depgrep',
    version='0.0.1',  # DO NOT EDIT THIS LINE MANUALLY. LET bump2version UTILITY DO IT
    author='Danny McDonald',
    author_email='mcddjx@gmail.com',
    description='Dependency parse searching',
    url='https://github.com/interrogator/depgrep',
    keywords='linguistics nlp dependencies conll',
    packages=['depgrep'],
    zip_safe=False,  # For mypy to be able to find the installed package
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    install_requires=['pandas',
                      'pyparsing',
                      'nltk'],
    python_requires='>=3.6',
    classifiers=['Topic :: Utilities'],
)
