import os
import numpy
import re
from setuptools import setup, find_packages, Extension
from sys import platform

def search(path, pattern):
    output = []
    for p in path.split(os.path.pathsep):
        files = os.listdir(p)
        for f in files:
            if re.match(pattern, f):
                output.append(f)
    return output

# set CC, CXX
if os.environ.get('CC') is None and os.environ.get('CXX') is None:
    path=os.getenv('PATH')
    gcc = search(path, '^gcc-\d+$')
    gxx = search(path, '^g\+\+-\d+$')
    if len(gcc) > 0 and len(gxx) > 0:
        print('Using versions: %s, %s' % (gcc[0], gxx[0]))
        os.environ['CC'] = gcc[0]
        os.environ['CXX'] = gxx[0]

# set extra_compile_args
gcc_version = re.findall('^gcc-(\d)', os.environ.get('CC'))
if len(gcc_version) == 1:
    gcc_version = int(gcc_version[0])
else:
    gcc_version = 0
if gcc_version > 4:
    # use openmp
    extra_compile_args = ['-fopenmp']
elif platform == 'darwin':
    extra_compile_args = ['-stdlib=libc++']
else:
    extra_compile_args = []
    
# set pool C++ extension
pool_module = Extension('pool', 
                        sources=['src/pool.cpp'],
                        language='C++',
                        include_dirs=[numpy.get_include()],
                        extra_compile_args=extra_compile_args,
                       )

# get long_description
with open('README.md', 'r') as f:
    long_description = f.read()

setup(name='rf_pool',
      version='0.1',
      author='Justin Theiss, Joel Bowen',
      author_email='theissjd@berkeley.edu',
      description='Receptive field pooling',
      long_description=long_description,
      url='https://github.com/jdtheiss/rf_pool',
      packages=find_packages(),
      ext_modules=[pool_module],
      )
