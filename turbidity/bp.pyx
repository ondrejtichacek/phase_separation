# distutils: language = c++
# cython: language_level=3

import numpy as np
cimport numpy as np
# import scipy.linalg.blas

import cython

# from cython.parallel import prange, parallel, threadid

# cimport openmp

# from libc.math cimport fabs, sqrt
# from libc.stdlib cimport malloc, free, calloc
# from libc.stdint cimport int64_t
# from libc.stdio cimport printf, fflush, stdout
# from libc.time cimport time, time_t
from libcpp cimport bool

# from svd3 import svd3, svd3pp

# cimport blis.cy

from common import timer

cdef extern from "bp_core.cpp":
    int bp_set(
        double lp,
        double lm,
        double l0,
        double Jp,
        double Jm,
        double Jpm,
        double J0,
        double J0p,
        double J0m,
        double dx,
        double dy,
        double X[],
        double Y[],
        double hX[],
        double hY[],
        double mp_[],
        double mm_[],
        double xp_[],
        double xm_[],
        int N
    ) nogil

    int hess_set(
        double dx,
        double dy,
        double hX_xy[],
        double hY_xy[],
        double hX_xyp[],
        double hY_xyp[],
        double hX_xpy[],
        double hY_xpy[],
        int sep[],
        int N
    ) nogil

    double calculate_err(
        int sep[],
        int sep_exp[],
        int is_on_boundary[],
        int N
    ) nogil

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef double core_set(
    double lp,
    double lm,
    double l0,
    double Jp,
    double Jm,
    double Jpm,
    double J0,
    double J0p,
    double J0m,
    double[:] x,
    double[:] y,
    double[:] hx_xy,
    double[:] hy_xy,
    double[:] hx_xpy,
    double[:] hy_xpy,
    double[:] hx_xyp,
    double[:] hy_xyp,
    double[:] mp_,
    double[:] mm_,
    double[:] xp_,
    double[:] xm_,
    int[:] sep,
    int[:] sep_exp,
    int[:] is_on_boundary,
    int N,
    ) nogil:

    cdef double dx = 1e-7
    cdef double dy = 1e-7

    cdef int s = 0

    s = bp_set(lp, lm, l0, Jp, Jm, Jpm, J0, J0p, J0m, 0, 0, &x[0], &y[0], &hx_xy[0], &hy_xy[0], &mp_[0], &mm_[0], &xp_[0], &xm_[0], N)
    if s > 0:
        return -1
    s = bp_set(lp, lm, l0, Jp, Jm, Jpm, J0, J0p, J0m, 0, dy, &x[0], &y[0], &hx_xyp[0], &hy_xyp[0], &mp_[0], &mm_[0], &xp_[0], &xm_[0], N)
    if s > 0:
        return -1
    s = bp_set(lp, lm, l0, Jp, Jm, Jpm, J0, J0p, J0m, dx, 0, &x[0], &y[0], &hx_xpy[0], &hy_xpy[0], &mp_[0], &mm_[0], &xp_[0], &xm_[0], N)
    if s > 0:
        return -1

    hess_set(dx, dy,
             &hx_xy[0], &hy_xy[0],
             &hx_xyp[0], &hy_xyp[0],
             &hx_xpy[0], &hy_xpy[0],
             &sep[0],
             N)


    cdef double err = 0.0
    err = calculate_err(
        &sep[0],
        &sep_exp[0],
        &is_on_boundary[0],
        N)

    return err

# @timer
@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def bp_main(
    double lp,
    double lm,
    double l0,
    double Jp,
    double Jm,
    double Jpm,
    double J0,
    double J0p,
    double J0m,
    double[:] x,
    double[:] y,
    double[:] hx_xy,
    double[:] hy_xy,
    double[:] hx_xpy,
    double[:] hy_xpy,
    double[:] hx_xyp,
    double[:] hy_xyp,
    double[:] mp_,
    double[:] mm_,
    double[:] xp_,
    double[:] xm_,
    int[:] sep,
    int[:] sep_exp,
    int[:] is_on_boundary,
    int N,
    ):

    return core_set(
        lp, lm, l0, Jp, Jm, Jpm, J0, J0p, J0m,
        x, y, hx_xy, hy_xy, hx_xpy,
        hy_xpy, hx_xyp, hy_xyp,
        mp_, mm_, xp_, xm_,
        sep, sep_exp, is_on_boundary, N)