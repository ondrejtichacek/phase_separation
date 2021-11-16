const int N = 200; // grid size NxN

double CP[N];
double CM[N];
double HH[2][N][N];
bool phase_sep[N][N];


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