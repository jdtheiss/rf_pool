from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(name='rf_pool',
      version='0.1',
      author='Justin Theiss, Joel Bowen',
      author_email='theissjd@berkeley.edu',
      description='Receptive field pooling',
      long_description=long_description,
      url='https://github.com/jdtheiss/rf_pool',
      packages=find_packages()
      )
