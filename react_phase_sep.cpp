#include "iostream"
#include "stdlib.h"
#include "math.h"
#include "time.h"
#include "fstream"
#include "vector"
#include "string"

using namespace std;

const int L = 100;    // size of the system: 2D square lattice  LxL periodic boundary conditions
const int q = 20;     // number of enzymes (length of the pathway)
double volume_frac;   // volume fraction of the solutes
double interaction;   // interaction between substrate and enzymes (uniform)

// global variables
double I[q + 1][q + 1]; // interaction matrix
double J[q + 1][q + 1]; // interaction matrix
int spin[L][L];         // lattice variables

double casual()
{ // a random number uniform in (0,1)
    long double x = double(random()) / (RAND_MAX + 1.0);
    return x;
}

double mean(const int array[], int size)
{
    double sum = 0.0;
    for (int i = 0; i < size; i++)
    {
        sum += array[i];
    }
    return sum / size;
}

double stdev(const int array[], int size)
{
    double mu = mean(array, size);
    double variance = 0.0;
    for (int i = 0; i < size; i++)
    {
        variance += pow(array[i] - mu, 2);
    }
    return sqrt(variance / size);
}

void init(const double conc)
{
    // initialization: filling interaction matrix & the lattice variables (uniform volume fraction)

    // interaction matrix
    for (int i = 0; i <= q; i++)
    {
        I[0][i] = 0;
        I[i][0] = 0; // interaction with solute (=0 default)
    }

    for (int i = 1; i <= q; i++)
        for (int j = 1; j <= q; j++)
            I[i][j] = 2*(1 - casual());

    for (int i = 1; i <= q; i++)
        for (int j = i; j <= q; j++)
            I[i][j] = 2 - casual();

    for (int i = 1; i <= q; i++)
        for (int j = i; j >= 1; j--)
            I[i][j] = 1 - casual();

    for (int i = 1; i <= q; i++)
    {
        for (int j = i; j <= q; j++)
        {
            if (i == j)
                I[i][j] = 0;
            if (i + 1 == j)
                I[i][j] = 2;
            //if (i - 1 == j)
            //    I[i][j] = 2;
        }
    }

    // interaction matrix
    for (int i = 0; i <= q; i++)
        J[0][i] = J[i][0] = 0; // interaction with solute (=0 default)

    for (int i = 1; i <= q; i++)
    {
        for (int j = i; j <= q; j++)
        {
            //double d = 2 - (j - i) / q;
            //J[i][j]=  2*(0.5-random(1));
            if (i == j)
                J[i][j] = 1; // self interaction  (=1 default)
            //else if (j - i <= 4)
            //    J[i][j] = 1.5;
            else if (j >= i + 1)
                // J[i][j] = 1; // cross interactions (=1 default)
                J[i][j] = 2*(1 - casual());
                // J[i][j] = d;
                // J[i][j] = 0;
            J[j][i] = J[i][j];
        }
    }

    // latice
    for (int i = 0; i <= L - 1; i++)
    { // filling the lattice with volume fraction "conc" at random (beta=0)
        for (int j = 0; j <= L - 1; j++)
        {
            double cas = casual();
            spin[i][j] = 0;
            for (int k = 1; k <= q; k++)
                if (cas >= (k - 1) * conc / q && cas < k * conc / q)
                    spin[i][j] = k;
        }
    }
}

double kawasaki(const double beta)
{ // Kawasaki Montecarlo:  exchange two particles position
    int i1 = int(casual() * L);
    int j1 = int(casual() * L);
    int ip = (i1 + 1) % L;
    int im = (L + i1 - 1) % L;
    int jp = (j1 + 1) % L;
    int jm = (L + j1 - 1) % L;

    int i2 = int(casual() * L);
    int j2 = int(casual() * L);
    int ip2 = (i2 + 1) % L;
    int im2 = (L + i2 - 1) % L;
    int jp2 = (j2 + 1) % L;
    int jm2 = (L + j2 - 1) % L;

    double energy_change = 0.0;

    if (spin[i1][j1] != spin[i2][j2])
    {
        double F1 = J[spin[i1][j1]][spin[i1][jm]] + J[spin[i1][j1]][spin[i1][jp]] + J[spin[i1][j1]][spin[im][j1]] + J[spin[i1][j1]][spin[ip][j1]];
        double F1n = J[spin[i1][j1]][spin[i2][jm2]] + J[spin[i1][j1]][spin[i2][jp2]] + J[spin[i1][j1]][spin[im2][j2]] + J[spin[i1][j1]][spin[ip2][j2]];
        double F2n = J[spin[i2][j2]][spin[i1][jm]] + J[spin[i2][j2]][spin[i1][jp]] + J[spin[i2][j2]][spin[im][j1]] + J[spin[i2][j2]][spin[ip][j1]];
        double F2 = J[spin[i2][j2]][spin[i2][jm2]] + J[spin[i2][j2]][spin[i2][jp2]] + J[spin[i2][j2]][spin[im2][j2]] + J[spin[i2][j2]][spin[ip2][j2]];

        double delta = F1 + F2 - F1n - F2n; // energy variation for the swap
        int news = 0;

        if (delta <= 0)
        {
            news = 1; // Always accept lower energy
        }
        else
        {
            double cas = casual();
            if (cas <= 1 / exp(beta * delta))
            {
                news = 1; // Metropolis rule
            }
        }

        if (news == 1)
        {
            int sp = spin[i1][j1];
            int sp2 = spin[i2][j2];
            spin[i2][j2] = sp;
            spin[i1][j1] = sp2;
        }

        energy_change = delta * news;
    }

    return energy_change;
}

double energy()
{
    double e = 0.0;

    // except for the boundary
    for (int i = 0; i < L; i++)
    {
        for (int j = 0; j < L; j++)
        {
            int ii = i+1; if (ii == L) ii = 0;
            int jj = j+1; if (jj == L) jj = 0;

            e += J[spin[i][j]][spin[ii][j]];
            e += J[spin[i][j]][spin[i][jj]];

            // not this becasue of symmetry
            //e += J[spin[i][j]][spin[i-1][j]];
            //e += J[spin[i][j]][spin[i][j-1]];
        }
    }
    return -e;
}

double sweep(const double beta)
{ // sweep over all system
    double energy_change = 0;

    for (int i = 0; i <= L * L - 1; i++)
        energy_change += kawasaki(beta);

    energy_change = energy();
    return energy_change;
}

int time_to_react(const double F, const double interaction)
{
    /* simulating a substrate entering and reacting,
    it returns the total time to go through the pathway,
    F is the attraction energy of the substrate with the enzymes (uniform) */
    int counter = 0;
    int pos[2];
    int tempo = 0;
    pos[0] = int(casual() * L);
    pos[1] = int(casual() * L);
    do
    {
        if (spin[pos[0]][pos[1]] == counter + 1)
        {
            counter++;
            //if (casual() < 1. / exp(1))
            //    counter++;
        }

        double cas = casual();
        int newpos[2];

        // random walk
        if (cas < 0.5)
            newpos[0] = pos[0] + 1;
        else
            newpos[0] = pos[0] - 1;

        if (newpos[0] < 0)
            newpos[0] = L - 1;
        else if (newpos[0] >= L)
            newpos[0] = 0;

        cas = casual();
        if (cas < 0.5)
            newpos[1] = pos[1] + 1;
        else
            newpos[1] = pos[1] - 1;

        if (newpos[1] < 0)
            newpos[1] = L - 1;
        else if (newpos[1] >= L)
            newpos[1] = 0;

        int okkei = 1;

        if (false)
        {
            // we go from K to 0, where K > 0
            if (spin[newpos[0]][newpos[1]] == 0 && spin[pos[0]][pos[1]] > 0)
            // if (spin[newpos[0]][newpos[1]] > 0 && spin[newpos[0]][newpos[1]] < counter + 1)
            // if (spin[newpos[0]][newpos[1]] > 0 && spin[newpos[0]][newpos[1]] != counter + 1)
            {
                // decide whether to accept step
                cas = casual();
                okkei = 0;
                if (cas < 1. / exp(F))
                    okkei = 1;
            }
        }
        else
        {
            cas = casual();
            okkei = 0;

            int i1 = counter;
            int j1 = spin[pos[0]][pos[1]];
            int i2 = counter;
            int j2 = spin[newpos[0]][newpos[1]];
            double T1 = I[i1][j1] - I[i2][j2];

            if (T1 < 0)
            {
                okkei = 1;
            }
            else
            {
                if (cas < 1. / exp(2*T1*interaction))
                    okkei = 1;
            }
        }

        if (okkei == 1)
        {
            pos[0] = newpos[0];
            pos[1] = newpos[1];
        }

        tempo++;

    } while (counter < q); // q is the chain length
    return tempo;
}

void load_lattice(const double beta)
{
    char fname[64];
    sprintf(fname, "lattice_%.1f.csv", beta);

    ifstream infile(fname);

    for (int i = 0; i < L; i++)
    {
        for (int j = 0; j < L; j++)
        {
            infile >> spin[i][j];
        }
    }
}

void save_lattice(const double beta)
{
    char fname[64];
    sprintf(fname, "lattice_%.1f.csv", beta);

    ofstream fout;
    fout.open(fname);

    for (int i = 0; i < L; i++)
    {
        for (int j = 0; j < L; j++)
        {
            fout << spin[i][j] << " ";
        }
        fout << endl;
    }
    fout.close();
}

void save_interaction()
{
    char fname1[100];
    char fname2[100];
    sprintf(fname1, "J.csv");
    sprintf(fname2, "I.csv");

    ofstream fout1;
    fout1.open(fname1);

    ofstream fout2;
    fout2.open(fname2);

    for (int i = 0; i < q+1; i++)
    {
        for (int j = 0; j < q+1; j++)
        {
            fout1 << J[i][j] << " ";
            fout2 << I[i][j] << " ";
        }
        fout1 << endl;
        fout2 << endl;
    }
    fout1.close();
    fout2.close();
}

void save_vector(const vector<int> &arr, char *fname)
{
    ofstream fout;
    fout.open(fname, ios_base::app);

    for (int i = 0; i < arr.size(); i++)
        fout << arr[i] << " ";
    
    fout << endl;
    fout.close();
}

void save_vector(const vector<double> &arr, char *fname)
{
    ofstream fout;
    fout.open(fname, ios_base::app);

    for (int i = 0; i < arr.size(); i++)
        fout << arr[i] << " ";
    
    fout << endl;
    fout.close();
}

void reaction(vector<double> &beta, const int sim_react)
{
    vector<int> tempo (sim_react);

    char fname[64];

    for (int j = 0; j < beta.size(); j++)
    {
        double b = beta[j];

        load_lattice(b);

        // cout << b << " " << endl;
        for (int i = 0; i < sim_react; i++)
        {
            // simulate the pathway
            tempo[i] = time_to_react(interaction * b, interaction);
        }

        sprintf(fname, "reaction_tempo.csv");
        save_vector(tempo, fname);
    }
}

void condensation(vector<double> &beta, const int sim_cond)
{
    vector<double> cond_energy (sim_cond);

    char fname[64];
    
    save_lattice(0.0);

    for (int j = 0; j < beta.size(); j++)
    {
        double b = beta[j];
        // cout << b << " " << endl;
        
        for (int i = 0; i < sim_cond; i++)
        {
            // simulate the condensation
            cond_energy[i] = sweep(b);
        }

        save_lattice(b);

        sprintf(fname, "cond_energy.csv");
        save_vector(cond_energy, fname);
    }
}

void init_beta(
    vector<double> &beta,
    const double betamin,
    const double betamax,
    const int betanum)
{
    for(int i = 1; i <= betanum; i++)
    {
        double b = betamin + (betamax - betamin) * double(i) / double(betanum); // inverse temperature (cooling)
        beta.push_back(b);
    }
}

int core(
    const double betamin,
    const double betamax,
    const int betanum,
    const int sim_cond,
    const int sim_react)
{
    srand(time(0));

    init(volume_frac);
    save_interaction();

    vector<double> beta;
    init_beta(beta, betamin, betamax, betanum);

    if (sim_cond > 0)
    {
        condensation(beta, sim_cond);
    }

    if (sim_react > 0)
    {
        reaction(beta, sim_react);
    }
    
    return 0;
}

int main(int argc, char **argv)
{
    interaction = 1;
    if (argc > 1)
        interaction = atof(argv[1]);

    volume_frac = 0.3;
    if (argc > 2)
        volume_frac = atof(argv[2]);

    double betamin = 0;
    if (argc > 3)
        betamin = atof(argv[3]);

    double betamax = 10;
    if (argc > 4)
        betamax = atof(argv[4]);

    int betanum = 100;
    if (argc > 5)
        betanum = atoi(argv[5]);

    int sim_cond = 1000;
    if (argc > 6)
        sim_cond = atoi(argv[6]);

    int sim_react = 1000;
    if (argc > 7)
        sim_react = atoi(argv[7]);
        

    return core(betamin, betamax, betanum, sim_cond, sim_react);
}