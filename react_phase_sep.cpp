#include "iostream"
#include "stdlib.h"
#include "math.h"
#include "time.h"
#include "fstream"
#include "vector"
#include "string"

using namespace std;

const int L = 100;        // size of the system: 2D square lattice  LxL periodic boundary conditions
const int q = 20;         // number of enzymes (length of the pathway)
double volume_frac = 0.3; // volume fraction of the solutes
double interaction = 1;   // interaction between substrate and enzymes (uniform)

// global variables
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

void init(double conc)
{
    // initialization: filling interaction matrix & the lattice variables (uniform volume fraction)

    // interaction matrix
    for (int i = 0; i <= q; i++)
        J[0][i] = J[i][0] = 1; // interaction with solute (=0 default)

    for (int i = 1; i <= q; i++)
    {
        for (int j = i; j <= q; j++)
        {
            //double d = 2 - (j - i) / q;
            //J[i][j]=  2*(0.5-random(1));
            if (i == j)
                J[i][j] = 0; // self interaction  (=1 default)
            else if (j - i <= 2)
                J[i][j] = 2;
            else if (j >= i + 1)
                // J[i][j] = 1; // cross interactions (=1 default)
                // J[i][j]=  2*(0.5-casual());
                // J[i][j] = d;
                J[i][j] = 0;
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

void kawasaki(double beta)
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

    if (spin[i1][j1] != spin[i2][j2])
    {
        double F1 = J[spin[i1][j1]][spin[i1][jm]] + J[spin[i1][j1]][spin[i1][jp]] + J[spin[i1][j1]][spin[im][j1]] + J[spin[i1][j1]][spin[ip][j1]];
        double F1n = J[spin[i1][j1]][spin[i2][jm2]] + J[spin[i1][j1]][spin[i2][jp2]] + J[spin[i1][j1]][spin[im2][j2]] + J[spin[i1][j1]][spin[ip2][j2]];
        double F2n = J[spin[i2][j2]][spin[i1][jm]] + J[spin[i2][j2]][spin[i1][jp]] + J[spin[i2][j2]][spin[im][j1]] + J[spin[i2][j2]][spin[ip][j1]];
        double F2 = J[spin[i2][j2]][spin[i2][jm2]] + J[spin[i2][j2]][spin[i2][jp2]] + J[spin[i2][j2]][spin[im2][j2]] + J[spin[i2][j2]][spin[ip2][j2]];

        double delta = F1 + F2 - F1n - F2n; // energy variation for the swap
        int news = -1;

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
    }
}

void sweep(double beta)
{ // sweep over all system
    for (int i = 0; i <= L * L - 1; i++)
        kawasaki(beta);
}

int time_to_react(double F)
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
            counter++;

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

        // we go from K to 0, where K > 0
        if (spin[newpos[0]][newpos[1]] == 0 && spin[pos[0]][pos[1]] > 0)
        {

            // decide whether to accept step
            cas = casual();
            okkei = 0;
            if (cas < 1. / exp(F))
                okkei = 1;
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

void save_lattice(double beta)
{
    char fname[100];
    sprintf(fname, "lattice_%.1f_%.1f_%.1f.csv", interaction, volume_frac, beta);

    ofstream fout;
    fout.open(fname);

    for (int i = 0; i < L; i++){
        for (int j = 0; j < L; j++){
            fout << spin[i][j] << " ";
        }
        fout << endl;
    }
    fout.close();
}

int core()
{
    srand(time(0));

    init(volume_frac);

    double betamin = 0;
    double betamax = 10;
    int sim_cond = 1000;
    int sim_react = 10;
    int number = 10;

    int tempo[sim_react];

    char fname[100];
    sprintf(fname, "out_%.1f_%.1f.csv", interaction, volume_frac);

    ofstream fout;
    fout.open(fname);

    for (int j = 1; j <= number; j++)
    {
        double beta = betamin + (betamax - betamin) * double(j) / double(number); // inverse temperature (cooling)
        for (int i = 0; i <= sim_cond; i++)
            sweep(beta); // simulate the condensation

        save_lattice(beta);
        
        for (int k = 0; k <= sim_react; k++)
        {
            tempo[k] = time_to_react(interaction * beta); // simulate the pathway
        }
        fout << beta << " "
             << mean(tempo, sim_react) << " "
             << stdev(tempo, sim_react) << " "
             << endl;
        // output: inverse temp vs average time to react
    }

    fout.close();

    return 0;
}

int main(int argc, char **argv)
{
    if (argc > 1)
        interaction = atof(argv[1]);
    if (argc > 2)
        volume_frac = atof(argv[2]);

    return core();
}