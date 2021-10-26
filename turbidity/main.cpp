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
// g++ BP_fit.cpp -O3
// ./a.out /path_to/mixed_atp.dat /path_to//mixed_atp.dat lp l0 lm Jp Jm Jpm J0 J0p J0m scale_x scale_y

using namespace std;

bool gnuplot_out = false;
bool err_out = true;

const int N = 200; // grid size NxN
double Hess[2][2];

double CP[N];
double CM[N];
double HH[2][N][N];
bool phase_sep[N][N];

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

double scale_x;
double scale_y;

double z = 6; // lattice connectivity
double dt = 0.1;
// double dt = 0.5;
// double dt = 1.0;
// double tol = 0.00000001;
double tol = 1e-8;

char *f_exp_ph_sep; //= 'exp/ph_sep_atp.dat';
char *f_exp_mixed;  //= 'exp/mixed_atp.dat';

const bool warm_start = true;
// const bool warm_start = false;

double uniform(const double a, const double b)
{ // a random number uniform in (a, b)
    return a + b * double(random()) / (RAND_MAX + 1.0);
}

int bp_grid()
{

    for (int i = 1; i < N; i++)
    {
        double cp = 1 * double(i) / double(N); // density of "plus" solute molecules
        CP[i] = cp;
    }
    for (int j = 1; j < N; j++)
    {
        double cm = 1 * double(j) / double(N); //density of "minus" solute molecules
        CM[j] = cm;
    }

    for (int i = 1; i < N; i++)
    {
        double cp = CP[i];
        for (int j = 1; j < N; j++)
        {
            double cm = CM[j];
            if (cp + cm < 1)
            {
                double fp, fm;
                double mp, mm;
                mp = cp;
                mm = cm;
                do
                {
                    //double Z = 1.+exp(mp) + exp(mm);
                    double Z = l0 + lp * exp(mp) + lm * exp(mm);

                    fp = mp + log(lp) - log(Z * cp);
                    fm = mm + log(lm) - log(Z * cm);

                    mp -= fp * dt;
                    mm -= fm * dt;

                } while (fabs(fp) + fabs(fm) > tol);

                double xp, xm;
                xp = cp;
                xm = cm;

                do
                {
                    // fp = xp - log((exp(Jp + mp - xp) + exp(Jpm + mm - xm) + 1.) / (exp(mp - xp) + exp(mm - xm) + 1.));
                    // fm = xm - log((exp(Jpm + mp - xp) + exp(Jm + mm - xm) + 1.) / (exp(mp - xp) + exp(mm - xm) + 1.));

                    // fp = xp - log(lp * (exp(Jp + mp - xp) + lm * exp(Jpm + mm - xm) + l0) / (lp * exp(mp - xp) + lm * exp(mm - xm) + l0));
                    // fm = xm - log(lp * (exp(Jpm + mp - xp) + lm * exp(Jm + mm - xm) + l0) / (lp * exp(mp - xp) + lm * exp(mm - xm) + l0));

                    fp = xp - log(lp * (exp(Jp + mp - xp) + lm * exp(Jpm + mm - xm) + l0 * exp(J0p)) / (lp * exp(J0p + mp - xp) + lm * exp(J0m + mm - xm) + l0 * exp(J0)));
                    fm = xm - log(lp * (exp(Jpm + mp - xp) + lm * exp(Jm + mm - xm) + l0 * exp(J0m)) / (lp * exp(J0p + mp - xp) + lm * exp(J0m + mm - xm) + l0 * exp(J0)));

                    xp -= fp * dt;
                    xm -= fm * dt;

                } while (fabs(fp) + fabs(fm) > tol);

                double hp, hm;
                hp = mp - z * xp;
                hm = mm - z * xm;

                HH[0][i][j] = hp;
                HH[1][i][j] = hm;

                // cout << cp << "  " << cm << "  " << hp << "  " << hm << endl; // write down the solution
            }
        }
    }

    for (int i = 1; i < N - 1; i++)
    {
        for (int j = 1; j < N - 1; j++)
        {
            if (CP[i] + CM[j] < 1)
            {
                Hess[0][0] = (HH[0][i + 1][j] - HH[0][i][j]) / (CP[i + 1] - CP[i]);
                Hess[1][1] = (HH[1][i][j + 1] - HH[1][i][j]) / (CM[j + 1] - CM[j]);
                Hess[1][0] = (HH[0][i][j + 1] - HH[0][i][j]) / (CM[j + 1] - CM[j]);
                Hess[0][1] = (HH[1][i + 1][j] - HH[1][i][j]) / (CP[i + 1] - CP[i]);

                double trace = Hess[0][0] + Hess[1][1];
                double deter = Hess[0][0] * Hess[1][1] - pow(0.5 * Hess[0][1] + 0.5 * Hess[1][0], 2);
                if (trace > 0 && deter > 0)
                {
                    phase_sep[i][j] = false;
                    // cout << CM[i] << "  " << CP[j] << endl; // write if positive definite (stable)
                }
                else
                {
                    phase_sep[i][j] = true;
                }
            }
        }
    }

    return 0;
}

// double mp_ = 0.0;
// double mm_ = 0.0;
// double xp_ = 0.0;
// double xm_ = 0.0;

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
    double l0expJ0 = l0 * exp(J0p);

    double loglp = log(lp);
    double loglm = log(lm);

    for (int i = 0; i < X.size(); i++)
    {
        double cp = X[i] * scale_x + dx;
        double cm = Y[i] * scale_y + dy;

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
        if (trace > 0 && deter > 0)
        {
            sep[i] = false;
            // cout << CM[i] << "  " << CP[j] << endl; // write if positive definite (stable)
        }
        else
        {
            sep[i] = true;
        }

        //sep[i] = double(random()) / (RAND_MAX + 1.0) > 0.5;
        //cout << "a";
    }
}

void load_exp_data(
    vector<double> &cp,
    vector<double> &cm,
    vector<bool> &sep)
{
    double p, m;

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

void load_mock_data(
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
            if (x + y < 1)
            {
                cp.push_back(x);
                cm.push_back(y);
                sep.push_back(true);
            }
        }
    }
}

void core_grid(
    const double lp,
    const double l0,
    const double lm,
    const double Jp,
    const double Jm,
    const double Jpm)
{
    bool solve_on_grid = false;
    bool gnuplot_out = false;
    bool err_out = true;

    if (solve_on_grid)
    {
        bp_grid();

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                if (phase_sep[i][j] == true)
                {
                    cout << CP[i] << " " << CM[j] << endl;
                }
            }
        }
    }
    else
    {
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

void init_set(Data &data, Mem &mem, const bool mock_data)
{
    if (mock_data)
        load_mock_data(data.x, data.y, data.sep_exp, 200);
    else
        load_exp_data(data.x, data.y, data.sep_exp);

    int n = data.x.size();

    for (int i = 0; i < n; i++)
    {
        // data.x[i] *= scale_x;
        //data.y[i] *= scale_y;

        if (data.x[i] + data.y[i] > 1)
        {
            //cout << data.x[i] << " + " << data.y[i] << " > 1" << endl;
            //throw std::runtime_error("cp + cm > 1");
        }
    }

    auto [min_x, max_x] = std::minmax_element(begin(data.x), end(data.x));
    if (DEBUG)
    {
        cout << "# x: [" << *min_x << ","<< *max_x << "]" << endl;
    }

    auto [min_y, max_y] = std::minmax_element(begin(data.y), end(data.y));
    if (DEBUG)
    {
        cout << "# y: [" << *min_y << ","<< *max_y << "]" << endl;
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
    // cout << Jpm << " ";

    double dx = 1e-5;// * scale_x;
    double dy = 1e-5;// * scale_y;

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
            err =  1.0;
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

void sweep()
{

    int NN = 10;

    Data data;
    Mem mem;

    init_set(data, mem, 0);

    for (int i = 0; i < NN; i++)
    {

        Jpm = 1.0 + 10.0 * double(i) / NN;

        core_set(data, mem);
    }
}

int main(int argc, char **argv)
{
    srand(time(0));

    int i = 0;

    i++;
    f_exp_ph_sep = argv[i];

    i++;
    f_exp_mixed = argv[i];

    // cout << f_exp_ph_sep;
    // cout << f_exp_mixed;

    i++;
    bool mock_data = 0;
    if (argc > i)
        mock_data = atoi(argv[i]);

    i++;
    int num_optimize = 0;
    if (argc > i)
        num_optimize = atoi(argv[i]);

    i++;
    scale_x = 1; //200/3;
    if (argc > i)
        scale_x = atof(argv[i]);

    i++;
    scale_y = 1; //1000/3;
    if (argc > i)
        scale_y = atof(argv[i]);

    i++;
    lp = 1;
    lp = 3.9366402;
    if (argc > i)
        lp = atof(argv[i]);

    i++;
    l0 = 1;
    l0 = 4.75065422;
    if (argc > i)
        l0 = atof(argv[i]);

    i++;
    lm = 1;
    lm = 5.33050119;
    if (argc > i)
        lm = atof(argv[i]);

    i++;
    Jp = -1;
    Jp = -4.82844051; 
    if (argc > i)
        Jp = atof(argv[i]);

    i++;
    Jm = -1;
    Jm = 0.27035404;
    if (argc > i)
        Jm = atof(argv[i]);

    i++;
    Jpm = 3;
    Jpm = 5.30866331;
    if (argc > i)
        Jpm = atof(argv[i]);
        

    i++;
    J0 = 0.1;
    J0 = -13.99622903;
    if (argc > i)
        J0 = atof(argv[i]);

    i++;
    J0p = 0.5;
    J0p = -7.17055996;  
    if (argc > i)
        J0p = atof(argv[i]);

    i++;
    J0m = 0.2;
    J0m = 5.06059504;
    if (argc > i)
        J0m = atof(argv[i]);

    if (DEBUG)
    {
        cout << "input par test:"
            << "scale_x " << scale_x << endl
            << "scale_y " << scale_y << endl
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

    // return 0;

    // --------------------------------------------------------------

    bool do_optimize = false;
    do_optimize = num_optimize > 0;

    Data data;
    Mem mem;

    if (!do_optimize)
    {
        // mock_data = true;
        // mock_data = false;
        
        gnuplot_out = true;
        err_out = false;
        
        init_set(data, mem, mock_data);
        core_set(data, mem);    

        return 0;
    }

    mock_data = false;
    gnuplot_out = false;
    err_out = true;

    init_set(data, mem, mock_data);
    
    double min_err, err;
    min_err = 1e9;


    double par_best[9+2];

    par_best[0] = lp;
    par_best[1] = lm;
    par_best[2] = l0;
    par_best[3] = Jp;
    par_best[4] = Jm;
    par_best[5] = Jpm;
    par_best[6] = J0;
    par_best[7] = J0p;
    par_best[8] = J0m;
    par_best[9] = scale_x;
    par_best[10] = scale_y;

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
            lp = par_best[0] + par_best[0]*s*uniform(-1, 1);
            lm = par_best[1] + par_best[1]*s*uniform(-1, 1);
            l0 = par_best[2] + par_best[2]*s*uniform(-1, 1);

            // // parameters (polymer interactions)
            Jp = par_best[3] + par_best[3]*s*uniform(-1, 1);
            Jm = par_best[4] + par_best[4]*s*uniform(-1, 1);
            Jpm = par_best[5] + par_best[5]*s*uniform(-1, 1);

            // // parameters (solvent interactions)
            J0 = par_best[6] + par_best[6]*s*uniform(-1, 1);
            J0p = par_best[7] + par_best[7]*s*uniform(-1, 1);
            J0m = par_best[8] + par_best[8]*s*uniform(-1, 1);

            // scale_x = par_best[9] + par_best[9]*s*uniform(-1, 1);
            // scale_y = par_best[10] + par_best[10]*s*uniform(-1, 1);
            // scale_x = uniform(1, 1);
            // scale_y = uniform(1, 1);
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
            par_best[9] = scale_x;
            par_best[10] = scale_y;

            if (DEBUG)
            {
                cout << min_err
                    << " ::" 
                    << " " << scale_x
                    << " " << scale_y
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
    scale_x = par_best[9];
    scale_y = par_best[10];

    cout << min_err << endl;

    if (DEBUG)
    {
        cout << "best par:"
            << " " << scale_x
            << " " << scale_y
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
            << "scale_x " << scale_x << endl
            << "scale_y " << scale_y << endl
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

    // gnuplot_out = true;
    // err_out = false;
    
    // core_set(data, mem); 

    return 0;
    
}