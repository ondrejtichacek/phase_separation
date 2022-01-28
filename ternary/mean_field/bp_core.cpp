#include "iostream"
#include "stdlib.h"
#include "math.h"
#include "time.h"
#include "fstream"
//#include <vector>
//#include <algorithm>
#include "string"
//#include <cassert>

#ifndef DEBUG
#define DEBUG false
#endif

using namespace std;

int bp_set(
    // polymer length "degeneracies"
    const double lp,
    const double lm,
    const double l0,
    // parameters (polymer interactions)
    const double Jp,
    const double Jm,
    const double Jpm,
    // parameters (solvent interactions)
    const double J0,
    const double J0p,
    const double J0m,
    //
    const double dx,
    const double dy,
    //
    const double sx,
    const double sy,
    //
    const double X[],
    const double Y[],
    //
    double hX[],
    double hY[],
    //
    double mp_[],
    double mm_[],
    double xp_[],
    double xm_[],
    //
    const double z, // lattice connectivity
    //
    const int N)
{
    // cout << N << endl;
    // cout << lp << endl;

    // const double z = 6;

    const double dt = 0.1;
    // const double dt = 0.5;
    // const double dt = 1.0;

    const double tol = 1e-8;

    const bool warm_start = true;
    // const bool warm_start = false;

    int it = 0;
    int k = 0;

    double l0expJ0 = l0 * exp(J0);
    double l0expJ0p = l0 * exp(J0p);
    double l0expJ0m = l0 * exp(J0m);

    double loglp = log(lp);
    double loglm = log(lm);

    for (int i = 0; i < N; i++)
    {
        // double cp = X[i] + dx;
        // double cm = Y[i] + dy;

        double cm = sx * (X[i] + dx);
        double cp = sy * (Y[i] + dy);

        double logcp = log(cp);
        double logcm = log(cm);

        if (true) //(cp + cm < 1)
        {
            double fp, fm;
            double mp, mm;

            if (warm_start)
            {
                mp = mp_[i];
                mm = mm_[i];
            }
            else
            {
                mp = cp;
                mm = cm;
            }

            k = 0;

            do
            {
                //double Z = 1.+exp(mp) + exp(mm);

                double logZ = log(l0 + lp * exp(mp) + lm * exp(mm));

                fp = mp + loglp - logZ - logcp;
                fm = mm + loglm - logZ - logcm;

                // double Z = l0 + lp * exp(mp) + lm * exp(mm);
                //
                // fp = mp + loglp - log(Z * cp);
                // fm = mm + loglm - log(Z * cm);

                mp -= fp * dt;
                mm -= fm * dt;

                k++;

                if (k > 1000000)
                {
                    cout << "fail at: " << cm << " " << cp << endl;
                    return 1;
                }

            } while (fabs(fp) + fabs(fm) > tol);
            //cout << k << endl;
            if (warm_start)
            {
                mp_[i] = mp + fp * dt;
                mm_[i] = mm + fm * dt;
            }

            double xp, xm;

            if (warm_start)
            {
                xp = xp_[i];
                xm = xm_[i];
            }
            else
            {
                xp = cp;
                xm = cm;
            }

            double J0p_mp = J0p + mp;
            double J0m_mm = J0m + mm;
            double Jp_mp = Jp + mp;
            double Jpm_mp = Jpm + mp;
            double Jpm_mm = Jpm + mm;
            double Jm_mm = Jm + mm;

            k = 0;

            do
            {
                it++;

                // fp = xp - log((exp(Jp + mp - xp) + exp(Jpm + mm - xm) + 1.) / (exp(mp - xp) + exp(mm - xm) + 1.));
                // fm = xm - log((exp(Jpm + mp - xp) + exp(Jm + mm - xm) + 1.) / (exp(mp - xp) + exp(mm - xm) + 1.));

                // fp = xp - log(lp * (exp(Jp + mp - xp) + lm * exp(Jpm + mm - xm) + l0) / (lp * exp(mp - xp) + lm * exp(mm - xm) + l0));
                // fm = xm - log(lp * (exp(Jpm + mp - xp) + lm * exp(Jm + mm - xm) + l0) / (lp * exp(mp - xp) + lm * exp(mm - xm) + l0));

                // fp = xp - log((lp * exp(Jp + mp - xp) + lm * exp(Jpm + mm - xm) + l0 * exp(J0p)) / (lp * exp(J0p + mp - xp) + lm * exp(J0m + mm - xm) + l0 * exp(J0)));
                // fm = xm - log((lp * exp(Jpm + mp - xp) + lm * exp(Jm + mm - xm) + l0 * exp(J0m)) / (lp * exp(J0p + mp - xp) + lm * exp(J0m + mm - xm) + l0 * exp(J0)));

                double d = (lp * exp(J0p_mp - xp) + lm * exp(J0m_mm - xm) + l0expJ0);

                fp = xp - log( (lp * exp(Jp_mp - xp) + lm * exp(Jpm_mm - xm) + l0expJ0p) / d);
                fm = xm - log( (lp * exp(Jpm_mp - xp) + lm * exp(Jm_mm - xm) + l0expJ0m) / d);

                xp -= fp * dt;
                xm -= fm * dt;

                k++;

                if (k > 1000000)
                {
                    cout << "fail at: " << cm << " " << cp << endl;
                    return 1;
                }

            } while (fabs(fp) + fabs(fm) > tol);
            // cout << k << endl;
            if (warm_start)
            {
                xp_[i] = xp + fp * dt;
                xm_[i] = xm + fm * dt;
            }

            double hp, hm;
            hp = mp - z * xp;
            hm = mm - z * xm;

            // hX[i] = hp;
            // hY[i] = hm;
            hX[i] = hm;
            hY[i] = hp;
        }
    }
    //cout << "#" << it << endl;
    return 0;
}

int hess_set(
    const double dx,
    const double dy,
    const double hX_xy[],
    const double hY_xy[],
    const double hX_xyp[],
    const double hY_xyp[],
    const double hX_xpy[],
    const double hY_xpy[],
    int sep[],
    const int N)
{
    double Hess[2][2];

    for (int i = 0; i < N; i++)
    {

        Hess[0][0] = (hX_xpy[i] - hX_xy[i]) / dx;
        Hess[1][1] = (hY_xyp[i] - hY_xy[i]) / dy;
        Hess[0][1] = (hX_xyp[i] - hX_xy[i]) / dy;
        Hess[1][0] = (hY_xpy[i] - hY_xy[i]) / dx;

        // cout << Hess[0][1] << " " << Hess[1][0] << endl;

        // if (fabs(Hess[0][1] - Hess[1][0]) > 5e-2 * (fabs(Hess[0][1]) + fabs(Hess[1][0])))
        // {
        //     if (DEBUG)
        //     {
        //         print_parameters();
        //         cout << Hess[0][1] << " " << Hess[1][0] << endl;
        //         throw std::runtime_error("Hessian must be symmetric");
        //     }
        //     else
        //     {
        //         return 1;
        //     }
        // }

        double trace = Hess[0][0] + Hess[1][1];
        // double deter = Hess[0][0] * Hess[1][1] - pow(0.5 * Hess[0][1] + 0.5 * Hess[1][0], 2);
        double deter = Hess[0][0] * Hess[1][1] - Hess[0][1] * Hess[1][0];

        // cout << trace << endl;
        // cout << deter << endl;

        if (trace > 0 && deter > 0) // if positive definite (stable)
        {
            // cout << 0;
            sep[i] = 0;
        }
        else
        {
            // cout << 1;
            sep[i] = 1;
        }
    }
    return 0;
}

double calculate_err(
    const int sep[],
    const int sep_exp[],
    const int is_on_boundary[],
    const int N,
    const double nmix,
    const double nsep
)
{
    double e = 0.0;
    double err = 0.0;
    bool all_sep = true;
    bool all_mix = true;
    bool all_sep_wrong = true;

    for (int i = 0; i < N; i++)
    {
        if (sep[i] == true)
        {
            if (sep[i] == sep_exp[i])
            {
                all_sep_wrong = false;
                break;
            }
        }
    }
    for (int i = 0; i < N; i++)
    {
        if (sep[i] == true)
        {
            all_mix = false;
            break;
        }
    }
    for (int i = 0; i < N; i++)
    {
        if (sep[i] == false)
        {
            all_sep = false;
            break;
        }
    }
    if (all_sep || all_mix || all_sep_wrong)
    {
        err = 1.0;
    }
    else
    {
        for (int i = 0; i < N; i++)
        {
            if (sep[i] != sep_exp[i])
            {
                if (is_on_boundary[i] == 1)
                {
                    e = 2.0;
                }
                else
                {
                    e = 1.0;
                }

                if (sep_exp[i] == 1)
                {
                    e /= nsep;
                }
                else if (sep_exp[i] == 0)
                {
                    e /= nmix;
                }

                err += e;
            }
        }
        err /= N;
    }
    return err * 100;
}

double calculate_continuous_err(
    const int sep[],
    const double sep_exp[],
    const int is_on_boundary[],
    const int N,
    const double nmix,
    const double nsep
)
{
    double e = 0.0;
    double err = 0.0;
    bool all_sep = true;
    bool all_mix = true;
    bool all_sep_wrong = false;

    for (int i = 0; i < N; i++)
    {
        if (sep[i] == true)
        {
            all_mix = false;
            break;
        }
    }
    for (int i = 0; i < N; i++)
    {
        if (sep[i] == false)
        {
            all_sep = false;
            break;
        }
    }
    if (all_sep || all_mix || all_sep_wrong)
    {
        err = 1.0;
    }
    else
    {
        for (int i = 0; i < N; i++)
        {
            e = sep_exp[i] - sep[i];

            if (is_on_boundary[i] == 1)
            {
                e *= 2.0;
            }

            // if (e > 0)
            // {
            //     e /= nsep;
            // }
            // else if (e < 0)
            // {
            //     e /= nmix;
            // }

            err += abs(e);
        }
        err /= N;
    }
    return err * 100;
}