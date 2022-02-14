#!/usr/bin/env sh

#
# This file is part of convwinograd
#
# An implementation of the Winograd-based convolution transform
#
# Copyright (C) 2021-22 Universitat Politècnica de València and
#                       Universitat Jaume I
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

VARIANT=WINGRD

NMIN=1
NMAX=1
NSTEP=1

KMIN=512
KMAX=512
KSTEP=64

CMIN=512
CMAX=512
CSTEP=64  

HMIN=4
HMAX=4
HSTEP=2   

WMIN=4
WMAX=4
WSTEP=2   

RMIN=3
RMAX=3
RSTEP=1   

SMIN=3 
SMAX=3 
SSTEP=1   

VPADMIN=1
VPADMAX=1
VPADSTEP=1

HPADMIN=1
HPADMAX=1
HPADSTEP=1   

VISUAL=0
TIMIN=5.0
TEST=T

./test_winograd \
  $VARIANT \
  $NMIN $NMAX $NSTEP \
  $KMIN $KMAX $KSTEP \
  $CMIN $CMAX $CSTEP \
  $HMIN $HMAX $HSTEP \
  $WMIN $WMAX $WSTEP \
  $RMIN $RMAX $RSTEP \
  $SMIN $SMAX $SSTEP \
  $VPADMIN $VPADMAX $VPADSTEP \
  $HPADMIN $HPADMAX $HPADSTEP \
  $VISUAL $TIMIN $TEST
