#include "iostream"
#include "stdlib.h"
#include "math.h"
#include "time.h"
#include "fstream"
#include <vector>
#include <algorithm>
#include "string"
#include <cassert>

#ifndef DEBUG
#define DEBUG false
#endif

// USAGE
// make or g++ main.cpp -O3 -std=c++17 -o turb_fit
// ./a.out path_to/ph_sep_atp.dat path_to/mixed_atp.dat grid_data optimize scale scale lp lm l0 Jp Jm Jpm J0 J0p J0m

using namespace std;

bool gnuplot_out = false;
bool err_out = true;

double Hess[2][2];

// polymer length "degeneracies"
double lp;
double lm;
double l0;

// parameters (polymer interactions)
double Jp;
double Jm;
double Jpm;

// parameters (solvent interactions)
double J0;
double J0p;
double J0m;

double limit_x;
double limit_y;

double z = 6; // lattice connectivity
double dt = 0.1;
// double dt = 0.5;
// double dt = 1.0;
// double tol = 0.00000001;
double tol = 1e-8;

char *f_exp_ph_sep; //= 'exp/ph_sep_atp.dat';
char *f_exp_mixed;  //= 'exp/mixed_atp.dat';

// const bool warm_start = true;
const bool warm_start = false;

double uniform(const double a, const double b)
{ // a random number uniform in (a, b)
    return a + b * double(random()) / (RAND_MAX + 1.0);
}

vector<double> mp_;
vector<double> mm_;
vector<double> xp_;
vector<double> xm_;

void bp_set(
    const double dx,
    const double dy,
    const vector<double> &X,
    const vector<double> &Y,
    vector<double> &hX,
    vector<double> &hY)
{
    int it = 0;
    double l0expJ0 = l0 * exp(J0);

    double loglp = log(lp);
    double loglm = log(lm);

    for (int i = 0; i < X.size(); i++)
    {
        double cp = X[i] + dx;
        double cm = Y[i] + dy;

        if (cp + cm < 1)
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

            do
            {
                //double Z = 1.+exp(mp) + exp(mm);
                double Z = l0 + lp * exp(mp) + lm * exp(mm);

                fp = mp + loglp - log(Z * cp);
                fm = mm + loglm - log(Z * cm);

                mp -= fp * dt;
                mm -= fm * dt;

            } while (fabs(fp) + fabs(fm) > tol);

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

            do
            {
                it++;
                // fp = xp - log((exp(Jp + mp - xp) + exp(Jpm + mm - xm) + 1.) / (exp(mp - xp) + exp(mm - xm) + 1.));
                // fm = xm - log((exp(Jpm + mp - xp) + exp(Jm + mm - xm) + 1.) / (exp(mp - xp) + exp(mm - xm) + 1.));

                // fp = xp - log(lp * (exp(Jp + mp - xp) + lm * exp(Jpm + mm - xm) + l0) / (lp * exp(mp - xp) + lm * exp(mm - xm) + l0));
                // fm = xm - log(lp * (exp(Jpm + mp - xp) + lm * exp(Jm + mm - xm) + l0) / (lp * exp(mp - xp) + lm * exp(mm - xm) + l0));

                // fp = xp - log(lp * (exp(Jp + mp - xp) + lm * exp(Jpm + mm - xm) + l0 * exp(J0p)) / (lp * exp(J0p + mp - xp) + lm * exp(J0m + mm - xm) + l0 * exp(J0)));
                // fm = xm - log(lp * (exp(Jpm + mp - xp) + lm * exp(Jm + mm - xm) + l0 * exp(J0m)) / (lp * exp(J0p + mp - xp) + lm * exp(J0m + mm - xm) + l0 * exp(J0)));

                double d = (lp * exp(J0p_mp - xp) + lm * exp(J0m_mm - xm) + l0expJ0);

                fp = xp - log(lp * (exp(Jp_mp - xp) + lm * exp(Jpm_mm - xm) + l0expJ0) / d);
                fm = xm - log(lp * (exp(Jpm_mp - xp) + lm * exp(Jm_mm - xm) + l0expJ0) / d);

                xp -= fp * dt;
                xm -= fm * dt;

            } while (fabs(fp) + fabs(fm) > tol);

            if (warm_start)
            {
                xp_[i] = xp + fp * dt;
                xm_[i] = xm + fm * dt;
            }

            double hp, hm;
            hp = mp - z * xp;
            hm = mm - z * xm;

            hX[i] = hp;
            hY[i] = hm;
        }
    }

    // cout << "#" << it << endl;
}

void hess_set(
    const double dx,
    const double dy,
    const vector<double> &hX_xy,
    const vector<double> &hY_xy,
    const vector<double> &hX_xyp,
    const vector<double> &hY_xyp,
    const vector<double> &hX_xpy,
    const vector<double> &hY_xpy,
    vector<bool> &sep)
{
    for (int i = 0; i < hX_xy.size(); i++)
    {

        Hess[0][0] = (hX_xpy[i] - hX_xy[i]) / dx;
        Hess[1][1] = (hY_xyp[i] - hY_xy[i]) / dy;
        Hess[0][1] = (hX_xyp[i] - hX_xy[i]) / dy;
        Hess[1][0] = (hY_xpy[i] - hY_xy[i]) / dx;

        double trace = Hess[0][0] + Hess[1][1];
        double deter = Hess[0][0] * Hess[1][1] - pow(0.5 * Hess[0][1] + 0.5 * Hess[1][0], 2);

        if (trace > 0 && deter > 0) // if positive definite (stable)
        {
            sep[i] = false;
        }
        else
        {
            sep[i] = true;
        }
    }
}

void load_exp_data(
    vector<double> &cp,
    vector<double> &cm,
    vector<bool> &sep)
{
    double p, m;

    // load phase separated
    ifstream f0(f_exp_ph_sep);
    if (f0.is_open())
    {
        while (f0.good())
        {
            f0 >> p >> m;
            if (p > 0 && m > 0)
            {
                cp.push_back(p);
                cm.push_back(m);
                sep.push_back(true);
            }
        }
    }
    else
    {
        throw std::runtime_error("error loading exp_ph_sep file");
    }

    // load phase separated
    ifstream f1(f_exp_mixed);
    if (f1.is_open())
    {
        while (f1.good())
        {
            f1 >> p >> m;
            if (p > 0 && m > 0)
            {
                cp.push_back(p);
                cm.push_back(m);
                sep.push_back(false);
            }
        }
    }
    else
    {
        throw std::runtime_error("error loading exp_mixed file");
    }

    if (DEBUG)
    {
        cout << "# " << sep.size() << endl;
    }
}

void load_use_grid_data(
    vector<double> &cp,
    vector<double> &cm,
    vector<bool> &sep,
    const int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double x = (i + 1.0) / n;
            double y = (j + 1.0) / n;

            x *= limit_x;
            y *= limit_y;

            if (x + y < 1)
            {
                cp.push_back(x);
                cm.push_back(y);
                sep.push_back(true);
            }
        }
    }
}

struct Data
{
    vector<double> x;
    vector<double> y;
    vector<bool> sep;
    vector<bool> sep_exp;
};

struct Mem
{
    vector<double> hx_xy;
    vector<double> hy_xy;

    vector<double> hx_xyp;
    vector<double> hy_xyp;

    vector<double> hx_xpy;
    vector<double> hy_xpy;

    vector<bool> boundary;
};

void init_set(Data &data, Mem &mem, const bool use_grid_data)
{
    if (use_grid_data)
        load_use_grid_data(data.x, data.y, data.sep_exp, 256);
    else
        load_exp_data(data.x, data.y, data.sep_exp);

    int n = data.x.size();

    for (int i = 0; i < n; i++)
    {
        if (data.x[i]  + data.y[i] > 1)
        {
            cout << data.x[i] << " + " << data.y[i] << " > 1" << endl;
            throw std::runtime_error("cp + cm > 1");
        }
    }

    auto [min_x, max_x] = std::minmax_element(begin(data.x), end(data.x));
    if (DEBUG)
    {
        cout << "# x: [" << *min_x << "," << *max_x << "]" << endl;
    }

    auto [min_y, max_y] = std::minmax_element(begin(data.y), end(data.y));
    if (DEBUG)
    {
        cout << "# y: [" << *min_y << "," << *max_y << "]" << endl;
    }

    for (int i = 0; i < n; i++)
    {
        data.sep.push_back(false);

        mem.hx_xy.push_back(0);
        mem.hy_xy.push_back(0);

        mem.hx_xyp.push_back(0);
        mem.hy_xyp.push_back(0);

        mem.hx_xpy.push_back(0);
        mem.hy_xpy.push_back(0);

        mp_.push_back(0);
        mm_.push_back(0);
        xp_.push_back(0);
        xm_.push_back(0);
    }

    for (int i = 0; i < n; i++)
    {
        if (data.x[i] < min_x[0] + 1e-5)
        {
            mem.boundary.push_back(true);
        }
        else if (data.x[i] > max_x[0] - 1e-5)
        {
            mem.boundary.push_back(true);
        }
        else if (data.y[i] < min_y[0] + 1e-5)
        {
            mem.boundary.push_back(true);
        }
        else if (data.y[i] > max_y[0] - 1e-5)
        {
            mem.boundary.push_back(true);
        }
        else
        {
            mem.boundary.push_back(false);
        }
    }
}
double core_set(Data &data, Mem &mem)
{
    double dx = 1e-6; // * limit_x;
    double dy = 1e-6; // * limit_y;

    bp_set(0, 0, data.x, data.y, mem.hx_xy, mem.hy_xy);
    bp_set(0, dy, data.x, data.y, mem.hx_xyp, mem.hy_xyp);
    bp_set(dx, 0, data.x, data.y, mem.hx_xpy, mem.hy_xpy);

    hess_set(dx, dy,
             mem.hx_xy, mem.hy_xy,
             mem.hx_xyp, mem.hy_xyp,
             mem.hx_xpy, mem.hy_xpy,
             data.sep);

    if (gnuplot_out)
    {
        ofstream fout_0;
        ofstream fout_1;
        fout_0.open("sep");
        fout_1.open("mix");

        for (int i = 0; i < data.x.size(); i++)
        {
            if (data.sep[i] == true)
                fout_0 << data.x[i] << " " << data.y[i] << endl;
            else
                fout_1 << data.x[i] << " " << data.y[i] << endl;
        }
        fout_0.close();
        fout_1.close();
    }

    if (err_out)
    {

        // assert(data.sep.size() == data.sep_exp.size());

        double err = 0;
        bool all_sep = true;
        bool all_mix = true;
        bool all_sep_wrong = true;

        for (int i = 0; i < data.sep.size(); i++)
        {
            if (data.sep[i] == true)
            {
                if (data.sep[i] == data.sep_exp[i])
                {
                    all_sep_wrong = false;
                    break;
                }
            }
        }
        for (int i = 0; i < data.sep.size(); i++)
        {
            if (data.sep[i] == true)
            {
                all_mix = false;
                break;
            }
        }
        for (int i = 0; i < data.sep.size(); i++)
        {
            if (data.sep[i] == false)
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
            for (int i = 0; i < data.sep.size(); i++)
            {
                if (data.sep[i] != data.sep_exp[i])
                {
                    if (mem.boundary[i] == true)
                    {
                        err += 10;
                    }
                    else
                    {
                        err++;
                    }
                }
            }
            err /= data.sep.size();
        }
        return err * 100;
    }

    return 0;
}

void optimize(Data &data, Mem &mem, const int num_optimize)
{
    gnuplot_out = false;
    err_out = true;

    init_set(data, mem, false);

    double min_err, err;
    min_err = 1e9;

    double par_best[9 + 2];

    par_best[0] = lp;
    par_best[1] = lm;
    par_best[2] = l0;
    par_best[3] = Jp;
    par_best[4] = Jm;
    par_best[5] = Jpm;
    par_best[6] = J0;
    par_best[7] = J0p;
    par_best[8] = J0m;
    par_best[9] = limit_x;
    par_best[10] = limit_y;

    double s = 1.00;

    for (int i = 0; i < num_optimize; i++)
    {
        if (DEBUG)
        {
            cout << ".";
            // cout << endl;
        }

        if (i > 0)
        {
            lp = par_best[0] + par_best[0] * s * uniform(-1, 1);
            lm = par_best[1] + par_best[1] * s * uniform(-1, 1);
            l0 = par_best[2] + par_best[2] * s * uniform(-1, 1);

            // // parameters (polymer interactions)
            Jp = par_best[3] + par_best[3] * s * uniform(-1, 1);
            Jm = par_best[4] + par_best[4] * s * uniform(-1, 1);
            Jpm = par_best[5] + par_best[5] * s * uniform(-1, 1);

            // // parameters (solvent interactions)
            J0 = par_best[6] + par_best[6] * s * uniform(-1, 1);
            J0p = par_best[7] + par_best[7] * s * uniform(-1, 1);
            J0m = par_best[8] + par_best[8] * s * uniform(-1, 1);

            // limit_x = par_best[9] + par_best[9]*s*uniform(-1, 1);
            // limit_y = par_best[10] + par_best[10]*s*uniform(-1, 1);
            // limit_x = uniform(1, 1);
            // limit_y = uniform(1, 1);
        }
        // cout << J0m << " ";

        err = core_set(data, mem);
        // cout << err <<  endl;
        if (err < min_err)
        {
            if (DEBUG)
            {
                cout << endl;
            }
            min_err = err;

            par_best[0] = lp;
            par_best[1] = lm;
            par_best[2] = l0;
            par_best[3] = Jp;
            par_best[4] = Jm;
            par_best[5] = Jpm;
            par_best[6] = J0;
            par_best[7] = J0p;
            par_best[8] = J0m;
            par_best[9] = limit_x;
            par_best[10] = limit_y;

            if (DEBUG)
            {
                cout << min_err
                     << " ::"
                     << " " << limit_x
                     << " " << limit_y
                     << " " << lp
                     << " " << lm
                     << " " << l0
                     << " " << Jp
                     << " " << Jm
                     << " " << Jpm
                     << " " << J0
                     << " " << J0p
                     << " " << J0m
                     << endl;
            }
        }
    }

    lp = par_best[0];
    lm = par_best[1];
    l0 = par_best[2];
    Jp = par_best[3];
    Jm = par_best[4];
    Jpm = par_best[5];
    J0 = par_best[6];
    J0p = par_best[7];
    J0m = par_best[8];
    limit_x = par_best[9];
    limit_y = par_best[10];

    cout << min_err << endl;

    if (DEBUG)
    {
        cout << "best par:"
             << " " << limit_x
             << " " << limit_y
             << " " << lp
             << " " << lm
             << " " << l0
             << " " << Jp
             << " " << Jm
             << " " << Jpm
             << " " << J0
             << " " << J0p
             << " " << J0m
             << endl;

        cout << "result:"
             << "limit_x " << limit_x << endl
             << "limit_y " << limit_y << endl
             << "lp " << lp << endl
             << "lm " << lm << endl
             << "l0 " << l0 << endl
             << "Jp " << Jp << endl
             << "Jm " << Jm << endl
             << "Jpm " << Jpm << endl
             << "J0 " << J0 << endl
             << "J0p " << J0p << endl
             << "J0m " << J0m << endl
             << endl;
    }
}

void print_parameters()
{
    cout << "input par test:"
            << "limit_x " << limit_x << endl
            << "limit_y " << limit_y << endl
            << "lp " << lp << endl
            << "lm " << lm << endl
            << "l0 " << l0 << endl
            << "Jp " << Jp << endl
            << "Jm " << Jm << endl
            << "Jpm " << Jpm << endl
            << "J0 " << J0 << endl
            << "J0p " << J0p << endl
            << "J0m " << J0m << endl
            << endl;
}

int main(int argc, char **argv)
{
    srand(time(0));

    int i = 0;

    i++;
    f_exp_ph_sep = argv[i];

    i++;
    f_exp_mixed = argv[i];

    i++;
    bool use_grid_data = 0;
    if (argc > i)
        use_grid_data = atoi(argv[i]);

    i++;
    int num_optimize = 0;
    if (argc > i)
        num_optimize = atoi(argv[i]);

    i++;
    limit_x = 1;
    if (argc > i)
        limit_x = atof(argv[i]);

    i++;
    limit_y = 1;
    if (argc > i)
        limit_y = atof(argv[i]);

    i++;
    lp = 1;
    if (argc > i)
        lp = atof(argv[i]);

    i++;
    lm = 1;
    if (argc > i)
        lm = atof(argv[i]);

    i++;
    l0 = 1;
    if (argc > i)
        l0 = atof(argv[i]);

    i++;
    Jp = -1;
    if (argc > i)
        Jp = atof(argv[i]);

    i++;
    Jm = -1;
    if (argc > i)
        Jm = atof(argv[i]);

    i++;
    Jpm = 3;
    if (argc > i)
        Jpm = atof(argv[i]);

    i++;
    J0 = 0.1;
    if (argc > i)
        J0 = atof(argv[i]);

    i++;
    J0p = 0.5;
    if (argc > i)
        J0p = atof(argv[i]);

    i++;
    J0m = 0.2;
    if (argc > i)
        J0m = atof(argv[i]);

    // --------------------------------------------------------------
    if (DEBUG)
        print_parameters();

    // --------------------------------------------------------------

    Data data;
    Mem mem;

    if (num_optimize == 1)
    {
        gnuplot_out = false;
        err_out = true;

        init_set(data, mem, use_grid_data);
        double err = core_set(data, mem);

        cout << err << endl;

        return 0;
    }
    else if (num_optimize > 1)
    {
        optimize(data, mem, num_optimize);
    }
    else
    {
        gnuplot_out = true;
        err_out = false;

        init_set(data, mem, use_grid_data);
        core_set(data, mem);

        return 0;
    }
}