convWinograd
============

The convWinograd library is an implementation of the Winograd-based convolution transform.


Compilation and installation
----------------------------

The next commands should be executed to compile and install the convWinograd library:

```shell
cd build
cmake [-D BLA_VENDOR=BLA_VENDOR] [-D CMAKE_INSTALL_PREFIX=INSTALL_PREFIX] ..
make                 # Alternatively:  cmake --build . --clean-first
make install         # Alternatively:  cmake --install . (this does not work with cmake older versions)
```

where ``BLA_VENDOR`` is used to force an specific BLAS vendor, as described next, and ``INSTALL_PREFIX`` is the prefix
PATH where ``lib/libconvWinograd.so`` should be installed. This last option is only required if the convWinograd library
should be installed on a prefix PATH different of ``/usr/local``.

As for the ``BLA_VENDOR`` option, the convWinograd library can use a CBLAS library or a bundled, but suboptimal,
implementation of the GEMM operation. To specify the CBLAS vendor to be used, the ``BLA_VENDOR`` option can be set to
one of:

* ``FLAME``: BLIS framework.
* ``Intel10_64ilp``: Intel MKL v10+ 64 bit, threaded code, ilp64 model.
* ``None``: The provided suboptimal GEMM.

If ``BLA_VENDOR`` is not set, all supported BLAS vendors will be tried in the order specified above.

Please note that in order to find a CBLAS library installed locally, with a prefix path different of the one where the
convWinograd library is going to be installed, the CBLAS library directory should be added to the ``LD_LIBRARY_PATH``
environment variable.

Be aware that a cmake system installation could favor those libraries installed by the distribution package manager,
thus ignoring those CBLAS libraries that have been manually installed. If this is the case, i.e., the
specified ``BLA_VENDOR`` option is ignored, a local version of cmake should be used (it can be installed
from <https://cmake.org/download/>).

Furthermore, if the CBLAS library is found but not its corresponding header (i.e., ``mkl.h`` of``cblas.h`` can not be
found), the ``-D CMAKE_PREFIX_PATH=CBLAS_INCLUDE_PREFIX`` should be added to the ``cmake ..``
command, where  ``CBLAS_INCLUDE_PREFIX`` is the directory where the required header is located.

For example, if the libconvWinograd should be installed under the ``~/opt/hpca_pydtnn`` prefix using the FLAME
framework, and the BLIS library is on the ``lib`` directory on the previous prefix, the next commands should be
executed:

```shell
cd build
cmake -D BLA_VENDOR=FLAME -D CMAKE_INSTALL_PREFIX=~/opt/hpca_pydtnn ..
make                 # Alternatively:  cmake --build . --clean-first
make install         # Alternatively:  cmake --install . (this does not work with cmake older versions)
```

Running the test
----------------

To run the included test, the ``-D COMPILE_TESTS=ON`` option must be passed in the configuration step and
the ``run_test_winograd`` target must be called in the build step. For example:

```shell
cd build
cmake -D BLA_VENDOR=FLAME -D CMAKE_INSTALL_PREFIX=~/opt/hpca_pydtnn -D COMPILE_TESTS=ON ..
make run_test_winograd  # Alternatively:  cmake --build . --target=run_test_winograd
```
