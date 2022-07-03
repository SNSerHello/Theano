制作wheel文件
============
$ python3 setup.py bdist_wheel --universal

为了支持GPU，需要编译安装libgpuarray，请参考：https://github.com/SNSerHello/libgpuarray

检查theano使用CPU还是GPU
=======================
from theano import function, config, shared, tensor as tt
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tt.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any(
    [
        isinstance(x.op, tt.elemwise.Elemwise) and ("Gpu" not in type(x.op).__name__)
        for x in f.maker.fgraph.toposort()
    ]
):
    print("Used the cpu")
else:
    print("Used the gpu")


参考：https://theano-pymc.readthedocs.io/en/latest/tutorial/using_gpu.html

在linux中搭建Anaconda3环境
=========================

1）访问：https://repo.anaconda.com/archive/
2）下载：https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh

$ wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh

因为这版本版本中使用python3.8.5，符合当前的需求，当前的最新版本为：https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh

$ wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh


在Linux中搭建Theano环境
======================

$ conda env create --file py38-theano.yaml

py38-theano.yaml配置如下：

name: py38-theano

# The conda channels to lookup the dependencies
channels:
  - anaconda
  - conda-forge

# The packages to install to the environment
dependencies:
  - conda-build
  - git
  - numpy
  - pytest
  - cython
  - cmake
  - bzip2
  - make
  - scipy
  - pillow
  - cudatoolkit-dev=11.3
  - cudnn
  - libgpuarray
  - ipython
  - pip
  - python=3.8
  - black
  - opencv
  - matplotlib
  - pandas
  - theano

.theanorc配置如下：

[global]
floatX = float32
device = cuda
optimizer_including = cudnn

[gcc]
# cxxflags = -I/media/samba/anaconda3/envs/py36-theano/include -L/media/samba/anaconda3/envs/py36-theano/lib -L/media/samba/anaconda3/envs/py36-theano/lib64 -L/usr/lib/x86_64-linux-gnu -lrt -pthread -lresolv
cxxflags = -I/media/samba/anaconda3/envs/py38-theano/include -L/media/samba/anaconda3/envs/py38-theano/lib -L/usr/lib/x86_64-linux-gnu -lrt -pthread -lresolv

[gpuarray]
preallocate = 0

[dnn]
enabled = True
# library_path = /media/samba/anaconda3/envs/py36-theano/lib
# include_path = /media/samba/anaconda3/envs/py36-theano/include
library_path = /media/samba/anaconda3/envs/py38-theano/lib
include_path = /media/samba/anaconda3/envs/py38-theano/include

[cuda]
cuda = /media/samba/anaconda3/envs/py38-theano/bin

[lib]
cnmem = 0.5

上面配置了python3.6与python3.8，在Ubuntu20.0LTS中，python3.6的动态编译很慢，不知道什么原因，所以建议使用python3.8。

附加py36-theano.yaml配置如下：

name: py36-theano

# The conda channels to lookup the dependencies
channels:
  - anaconda
  - conda-forge

# The packages to install to the environment
dependencies:
  - conda-build
  - git
  - numpy
  - pytest
  - cython
  - cmake
  - bzip2
  - make
  - scipy
  - pillow
  - cudatoolkit-dev=9.2
  - cudnn=7.1
  - ipython
  - jedi
  - pip
  - python=3.6
  - black
  - opencv
  - matplotlib
  - pandas
  - theano

为不同conda环境设置环境变量
=========================

$ mkdir -p $CONDA_PREFIX/etc/conda/activate.d

$ nano $CONDA_PREFIX/etc/conda/activate.d/env_vars.h

CUDA_HOME=$CONDA_PREFIX

export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

$ mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

$ nano $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.h

export LD_LIBRARY_PATH=`echo $LD_LIBRARY_PATH | cut -d : -f 2-`


============================================================================================================
MILA will stop developing Theano: https://groups.google.com/d/msg/theano-users/7Poq8BZutbY/rNCIfvAEAwAJ

The PyMC developers have forked Theano to a new project called Aesara that is being actively developed: https://github.com/aesara-devs/aesara
============================================================================================================


To install the package, see this page:
   http://deeplearning.net/software/theano/install.html

For the documentation, see the project website:
   http://deeplearning.net/software/theano/

Related Projects:
   https://github.com/Theano/Theano/wiki/Related-projects

It is recommended that you look at the documentation on the website, as it will be more current than the documentation included with the package.

In order to build the documentation yourself, you will need sphinx. Issue the following command:

::

   python ./doc/scripts/docgen.py

Documentation is built into ``html/``

The PDF of the documentation can be found at ``html/theano.pdf``

================
DIRECTORY LAYOUT
================

``Theano`` (current directory) is the distribution directory.

* ``Theano/theano`` contains the package
* ``Theano/theano`` has several submodules:
 
  * ``gof`` + ``compile`` are the core
  * ``scalar`` depends upon core
  * ``tensor`` depends upon ``scalar``
  * ``sparse`` depends upon ``tensor``
  * ``sandbox`` can depend on everything else

* ``Theano/examples`` are copies of the example found on the wiki
* ``Theano/benchmark`` and ``Theano/examples`` are in the distribution, but not in
  the Python package
* ``Theano/bin`` contains executable scripts that are copied to the bin folder
  when the Python package is installed
* Tests are distributed and are part of the package, i.e. fall in
  the appropriate submodules
* ``Theano/doc`` contains files and scripts used to generate the documentation
* ``Theano/html`` is where the documentation will be generated
