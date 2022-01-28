#include "iostream"
#include "stdlib.h"
#include "math.h"
#include "time.h"
#include "fstream"
#include "vector"
#include "string"
#include "assert.h"

using namespace std;

#ifndef DO_REACTION
#define DO_REACTION true
#endif

#ifndef INTERACTION_WALK
#define INTERACTION_WALK true
#endif

#ifndef LOG_CONCENTRATION
#define LOG_CONCENTRATION false
#endif

#ifndef EXTENDED_NEIGHBORHOOD
#define EXTENDED_NEIGHBORHOOD false
#endif

#ifndef OPEN_SYSTEM
#define OPEN_SYSTEM false
#endif

#ifndef VERTICAL_DIFFUSION_OF_PRODUCT
#define VERTICAL_DIFFUSION_OF_PRODUCT false
#endif

#ifndef TRANSFORM_PRODUCT_TO_SOLVENT
#define TRANSFORM_PRODUCT_TO_SOLVENT false
#endif

#ifndef LATTICE_SIZE
#define LATTICE_SIZE 100
#endif

#ifndef NUM_COMPONENTS
#define NUM_COMPONENTS 20
#endif

const int L = LATTICE_SIZE;   // size of the system: 2D square lattice  LxL periodic boundary conditions
const int q = NUM_COMPONENTS; // number of enzymes (length of the pathway)

// global variables
double I[q + 1][q + 1]; // interaction matrix
double J[q + 1][q + 1]; // interaction matrix

int8_t spin[L][L]; // lattice variables
double ener[L][L]; // energy at lattice points
bool ener_current[L][L];

int substrate[q + 1][L][L];
uint l_react[q + 1][L][L];

double volume_frac; // volume fraction of the solutes
// double interaction; // interaction between substrate and enzymes (uniform)

const int diffusion_kernel_size = 3;
double diffusion_kernel[3][3];

void init_diffusion_kernel()
{
//   1/120, 1/60, 1/120
//   1/60,  9/10, 1/60
//   1/120, 1/60, 1/120

diffusion_kernel[0][0] = 1/120;
diffusion_kernel[0][2] = 1/120;
diffusion_kernel[2][0] = 1/120;
diffusion_kernel[2][2] = 1/120;

diffusion_kernel[0][1] = 1/60;
diffusion_kernel[1][0] = 1/60;
diffusion_kernel[1][2] = 1/60;
diffusion_kernel[2][1] = 1/60;

diffusion_kernel[1][1] = 9/10;

}

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

void convolution_2d_pbc(int a, int b, int k) {

// A = B @ M

// find center position of kernel (half of kernel size)
// assuming kernel size is odd
int c = (diffusion_kernel_size - 1) / 2;

for (int i = 0; i < L; i++)              // rows
{
    for (int j = 0; j < L; j++)          // columns
    {
        for (int m = 0; m < diffusion_kernel_size; m++) // kernel rows
        {
            for (int n = 0; n < diffusion_kernel_size; n++) // kernel columns
            {
                int ii = (i + (m - c)) % L;
                int jj = (j + (n - c)) % L;

                // substrate[a][k][i][j] += substrate[b][k][ii][jj] * diffusion_kernel[m][n];
            }
        }
    }
}
}

void init_lattice(const double conc)
{
    // initialization: filling lattice variable (uniform volume fraction)
    // latice
    for (int i = 0; i <= L - 1; i++)
    { // filling the lattice with volume fraction "conc" at random (beta=0)
        for (int j = 0; j <= L - 1; j++)
        {
            ener[i][j] = 0;
            ener_current[i][j] = 0;

            double cas = casual();
            spin[i][j] = 0;
            for (int k = 1; k <= q; k++)
                if (cas >= (k - 1) * conc / q && cas < k * conc / q)
                    spin[i][j] = k;
        }
    }
}

double metropolis(const double beta, const double mu)
{
    int i = int(casual() * L); // orig x
    int j = int(casual() * L); // orig y
    int ip = (i + 1) % L;     //      x + 1
    int im = (L + i - 1) % L; //      x - 1
    int jp = (j + 1) % L;     //      y + 1
    int jm = (L + j - 1) % L; //      y - 1

    double delta[q + 1];
    double x[q + 1];
    double P[q + 1];

    // z ... partition sum
    double z = 0.0;

    // k ... "color"
    for (int k = 0; k < q + 1; k++){

        // F ... interaction energy with neighbors
        double F = (J[k][spin[i][jm]]   // down interaction
                  + J[k][spin[i][jp]]   // up
                  + J[k][spin[im][j]]   // left
                  + J[k][spin[ip][j]]); // right

        delta[k] = -F;

        if (k != 0){
            delta[k] += mu;
        }

        x[k] = exp(-beta * delta[k]);
        z += x[k];
    }
    
    // cumulative probability
    double P_cum = 0.0;

    // random number
    double cas = casual();

    for (int k = 0; k < q + 1; k++){
        P[k] = x[k] / z;
        P_cum += P[k];
        if (cas < P_cum)
        {
            spin[i][j] = k;
            return 0.0;
        }
    }

}

// double dEnergy(
//     double dE[],
//     const int s, 
//     const int i, const int j,
//     const int im, const int jm,
//     const int ip, const int jp)
// {
//     dE[0] = J[s][spin[i][jm]]; // down interaction
//     dE[1] = J[s][spin[i][jp]]; // up
//     dE[2] = J[s][spin[im][j]]; // left
//     dE[3] = J[s][spin[ip][j]]; // right

//     return dE[0] + dE[1] + dE[2] + dE[3]; 
// }

// void updateEnergy(
//     double dE[],
//     const int i, const int j,
//     const int im, const int jm,
//     const int ip, const int jp)
// {
//     ener[i][jm] += dE[0];
//     ener[i][jp] += dE[1];
//     ener[im][j] += dE[2];
//     ener[ip][j] += dE[3];
// }

double dEnergy(
    const int s, 
    const int i, const int j,
    const int im, const int jm,
    const int ip, const int jp)
{
    return (J[s][spin[i][jm]]   // down interaction
          + J[s][spin[i][jp]]   // up
          + J[s][spin[im][j]]   // left
          + J[s][spin[ip][j]]); // right
}

double kawasaki(const double beta, const double Ainv, const int s_A)
{ // Kawasaki Montecarlo:  exchange two particles position
    int i1 = int(casual() * L); // orig x
    int j1 = int(casual() * L); // orig y
    int ip1 = (i1 + 1) % L;     //      x + 1
    int im1 = (L + i1 - 1) % L; //      x - 1
    int jp1 = (j1 + 1) % L;     //      y + 1
    int jm1 = (L + j1 - 1) % L; //      y - 1

    int i2 = int(casual() * L); //  new x
    int j2 = int(casual() * L); //  new y
    int ip2 = (i2 + 1) % L;
    int im2 = (L + i2 - 1) % L;
    int jp2 = (j2 + 1) % L;
    int jm2 = (L + j2 - 1) % L;

    double energy_change = 0.0;

    // double dG1[4];
    // double dG2[4];

    if (spin[i1][j1] != spin[i2][j2])
    {

        if (   (j1 == j2 && ( i1 == ip2 || i1 == im2 ))
            || (i1 == i2 && ( j1 == jp2 || j1 == jm2 )))
        {
            return 0.0;
        }

        double F1;
        // energy contribution of spot 1 ORIG
        if (false) //(ener_current[i1][j1])
        {
            F1 = ener[i1][j1];
        }
        else
        {
            F1 = dEnergy(spin[i1][j1], i1, j1, im1, jm1, ip1, jp1);
            // double F1 = (J[spin[i1][j1]][spin[i1][jm1]]   // down interaction
            //            + J[spin[i1][j1]][spin[i1][jp1]]   // up
            //            + J[spin[i1][j1]][spin[im1][j1]]   // left
            //            + J[spin[i1][j1]][spin[ip1][j1]]); // right
        }

        double F2;
        // energy contribution of spot 2 ORIG
        if (false) //(ener_current[i2][j2])
        {
            F2 = ener[i2][j2];
        }
        else
        {
            F2 = dEnergy(spin[i2][j2], i2, j2, im2, jm2, ip2, jp2);
            // double F2 = (J[spin[i2][j2]][spin[i2][jm2]]   
            //            + J[spin[i2][j2]][spin[i2][jp2]]
            //            + J[spin[i2][j2]][spin[im2][j2]]
            //            + J[spin[i2][j2]][spin[ip2][j2]]);
        }

        // energy contribution of spot 1 NEW
        double G1 = dEnergy(spin[i1][j1], i2, j2, im2, jm2, ip2, jp2);
        // double G1 = (J[spin[i1][j1]][spin[i2][jm2]]
        //            + J[spin[i1][j1]][spin[i2][jp2]]
        //            + J[spin[i1][j1]][spin[im2][j2]]
        //            + J[spin[i1][j1]][spin[ip2][j2]]);

        // energy contribution of spot 2 NEW
        double G2 = dEnergy(spin[i2][j2], i1, j1, im1, jm1, ip1, jp1);
        // double G2 = (J[spin[i2][j2]][spin[i1][jm1]]
        //            + J[spin[i2][j2]][spin[i1][jp1]]
        //            + J[spin[i2][j2]][spin[im1][j1]]
        //            + J[spin[i2][j2]][spin[ip1][j1]]);

        if (EXTENDED_NEIGHBORHOOD == true)
        {
            F1 += (J[spin[i1][j1]][spin[im1][jm1]]
                 + J[spin[i1][j1]][spin[ip1][jp1]]
                 + J[spin[i1][j1]][spin[im1][jp1]]
                 + J[spin[i1][j1]][spin[ip1][jm1]]);

            F2 += (J[spin[i2][j2]][spin[im2][jm2]]
                 + J[spin[i2][j2]][spin[ip2][jp2]]
                 + J[spin[i2][j2]][spin[im2][jp2]]
                 + J[spin[i2][j2]][spin[ip2][jm2]]);

            G1 += (J[spin[i1][j1]][spin[im2][jm2]]
                 + J[spin[i1][j1]][spin[ip2][jp2]]
                 + J[spin[i1][j1]][spin[im2][jp2]]
                 + J[spin[i1][j1]][spin[ip2][jm2]]);

            G2 += (J[spin[i2][j2]][spin[im1][jm1]]
                 + J[spin[i2][j2]][spin[ip1][jp1]]
                 + J[spin[i2][j2]][spin[im1][jp1]]
                 + J[spin[i2][j2]][spin[ip1][jm1]]);
        }

        double fac = (1 + substrate[s_A][i1][j1] * Ainv);

        F1 *= fac;
        F2 *= fac;

        double delta = F1 + F2 - G1 - G2; // energy variation for the swap

        if (EXTENDED_NEIGHBORHOOD == true)
        {
            delta /= 2;
        }

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
            int sp1 = spin[i1][j1];
            int sp2 = spin[i2][j2];

            spin[i2][j2] = sp1;
            spin[i1][j1] = sp2;

            ener[i1][j1] = G2;
            ener[i2][j2] = G1;

            ener_current[i1][j1] = true;
            ener_current[i2][j2] = true;

            ener_current[i1][jm1] = false;
            ener_current[i1][jp1] = false;
            ener_current[im1][j1] = false;
            ener_current[ip1][j1] = false;

            ener_current[i2][jm2] = false;
            ener_current[i2][jp2] = false;
            ener_current[im2][j2] = false;
            ener_current[ip2][j2] = false;
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
            int ii = i + 1;
            if (ii == L)
                ii = 0;
            int jj = j + 1;
            if (jj == L)
                jj = 0;

            e += J[spin[i][j]][spin[ii][j]];
            e += J[spin[i][j]][spin[i][jj]];

            // not this becasue of symmetry
            //e += J[spin[i][j]][spin[i-1][j]];
            //e += J[spin[i][j]][spin[i][j-1]];
        }
    }
    return -e;
}

void reaction_new(const double beta)
{
    int i = int(casual() * L); // orig x
    int j = int(casual() * L); // orig y

    int cat = spin[i][j];
    if (cat > 0){
        // reaction
        substrate[cat][i][j] += substrate[cat - 1][i][j];
        substrate[cat - 1][i][j] = 0;
    }
}

int pos[5][2];

void lateral_diffusion(const double beta, const double inter)
{
    int i = int(casual() * L); // orig x
    int j = int(casual() * L); // orig y

    int ip = (i + 1) % L;     //      x + 1
    int im = (L + i - 1) % L; //      x - 1
    int jp = (j + 1) % L;     //      y + 1
    int jm = (L + j - 1) % L; //      y - 1

    pos[0][0] = i;
    pos[0][1] = j;

    pos[1][0] = i;
    pos[1][1] = jm;

    pos[2][0] = i;
    pos[2][1] = jp;

    pos[3][0] = im;
    pos[3][1] = j;

    pos[4][0] = ip;
    pos[4][1] = j;

    double delta[5];
    double x[5];
    double P[5];
    int dS_int[5];

    for (int s = 0; s < q + 1; s++)
    {

        delta[0] = I[s][spin[i][j]];
        delta[1] = I[s][spin[i][jm]]; // down interaction
        delta[2] = I[s][spin[i][jp]]; // up
        delta[3] = I[s][spin[im][j]]; // left
        delta[4] = I[s][spin[ip][j]]; // right

        // z ... partition sum
        double z = 0.0;

        for (int k = 0; k < 5; k ++)
        {
            delta[k] = - I[s][spin[pos[k][0]][pos[k][1]]];

            x[k] = exp(-beta * inter * delta[k]);
            z += x[k];
        }

        int S = substrate[s][i][j];

        assert(substrate[s][i][j] >= 0);

        for (int k = 0; k < 5; k++)
        {
            P[k] = x[k] / z;

            assert(P[k] >= 0);
            assert(P[k] <= 1);

            double dS = S * P[k];

            if (dS >= 1)
            {
                dS_int[k] = floor(dS);
                substrate[s][i][j] -= dS_int[k];
            }
            else
            {
                dS_int[k] = 0;
            }
        }

        S = substrate[s][i][j];

        assert(substrate[s][i][j] >= 0);
        assert(substrate[s][i][j] <= 5);

        for (int l = 0; l < S; l++)
        {

            // random number
            double cas = casual();
            
            // cumulative probability
            double P_cum = 0.0;

            for (int k = 0; k < 5; k++)
            {
                P_cum += P[k];
                if (cas < P_cum)
                {
                    dS_int[k]++;
                    substrate[s][i][j]--;
                    break;
                }
            }
        }

        if (substrate[s][i][j] != 0)
        {
            cout << substrate[s][i][j];
        }

        assert(substrate[s][i][j] == 0);

        for (int k = 0; k < 5; k++)
        {
            substrate[s][pos[k][0]][pos[k][1]] += dS_int[k];
        }

        assert(substrate[s][i][j] >= 0);

    }
}

void product_to_solvent(const int num_convert, const int i, const int j)
{
    if (substrate[q][i][j] >= num_convert){
        substrate[q][i][j] -= num_convert;

        int cat = spin[i][j];

        spin[i][j] = 0;

        if (cat > 0){
            while (true)
            {
                int k = int(casual() * L); // orig x
                int l = int(casual() * L); // orig y

                if (spin[k][l] == 0)
                {
                    double dx = abs(i - k);
                    double dy = abs(j - l);

                    dx = min(dx, abs(dx - L));
                    dy = min(dy, abs(dy - L));

                    double dist = sqrt(dx*dx + dy*dy);
                    dist = dist * sqrt(2) / L;

                    double p = (1 - dist);
                    p = p * p;

                    if (casual() < p){
                        spin[k][l] = cat;
                        break;
                    }
                }
            }
        }
    }

}

void vertical_diffusion(
    const double a, // influx rate
    const double b, // outflux rate
    const int i,    // x-ind
    const int j,    // y-ind
    const int k     // component
    )
{
    // bulk diffusion only with initial substrate

    int C = substrate[k][i][j];
    double dC = a - b * C;

    if (C + dC <= 0){
        substrate[k][i][j] = 0;
        return;
    }

    if (dC > 0)
    {
        C += floor(dC);
        dC -= floor(dC);
    }
    else
    {
        C += ceil(dC);
        dC -= ceil(dC);
    }

    if (C > 0){
        if (casual() < dC){
            C--;
        }
    }

    substrate[k][i][j] = C;
}


double sweep(
    const double beta,
    const double mu, 
    const double alpha)
{    
    double energy_change = 0;

    double scale = 1000;

    // --------------------------------------------------------------
    // diffusion of substrate #0 from the bulk to the membrane
    double influx_rate = alpha * scale;
    double outflux_rate = 0.1;

    if (DO_REACTION)
    {
        for (int i = 0; i < L; i++)
        {
            for (int j = 0; j < L; j++)
            {
                vertical_diffusion(influx_rate, outflux_rate, i, j, 0);
            }
        }
    }
    // --------------------------------------------------------------

    int K = 1; // number of reaction-diffusion loops

    int substrate_bind_index = q-1; // if q ... product
    double substrate_bind_scale;
    if (VERTICAL_DIFFUSION_OF_PRODUCT)
        substrate_bind_scale = 0.1 / scale;
    else
        substrate_bind_scale = 0; // to turn off

    for (int it = 0; it < L * L; it++)
    {
        if (OPEN_SYSTEM)
            energy_change += metropolis(beta, mu);
        else
            energy_change += kawasaki(beta, substrate_bind_scale, substrate_bind_index);

        if (DO_REACTION)
        {
            for (int k = 0; k < K; k++)
            {
                double inter = 1;
                lateral_diffusion(beta, inter);

                reaction_new(beta);
            }
        }
    }

    // --------------------------------------------------------------
    // diffusion of product (substrate #q) from the membrane to the bulk

    if (DO_REACTION && VERTICAL_DIFFUSION_OF_PRODUCT)
    {
        influx_rate = 0;
        outflux_rate = 0.3;

        for (int i = 0; i < L; i++)
        {
            for (int j = 0; j < L; j++)
            {
                vertical_diffusion(influx_rate, outflux_rate, i, j, q);
            }
        }
    }

    // --------------------------------------------------------------
    // product is transformed to solvent

    if (DO_REACTION && TRANSFORM_PRODUCT_TO_SOLVENT)
    {
        int num_convert = 20 * scale;

        for (int it = 0; it < L*L; it++)
        {
            int i = int(casual() * L);
            int j = int(casual() * L);

            product_to_solvent(num_convert, i, j);
        }
    }
    // --------------------------------------------------------------

    energy_change = energy();
    return energy_change;
}

int time_to_react(const double F, const double interaction, const int react_counter)
{
    /* simulating a substrate entering and reacting,
    it returns the total time to go through the pathway,
    F is the attraction energy of the substrate with the enzymes */
    int counter = 0;
    
    int pos[2];
    int newpos[2];

    int tempo = 0;

    pos[0] = int(casual() * L);
    pos[1] = int(casual() * L);

    do
    {
        // l_react[pos[0]][pos[1]][counter] += 1;

        if (spin[pos[0]][pos[1]] == counter + 1)
        {
            // if counter < 5)
            //     l_react[pos[0]][pos[1]][counter] / react_counter

            counter++;
            //if (casual() < 1. / exp(1))
            //    counter++;
        }

        // random walk
        int s = 1;
        if (casual() < 0.5)
            s = -1;

        if (casual() < 0.5)
        {
            newpos[0] = pos[0] + s;
            newpos[1] = pos[1];
        }
        else
        {
            newpos[0] = pos[0];
            newpos[1] = pos[1] + s;
        }

        // periodic boundary
        if (newpos[0] < 0)
            newpos[0] = L - 1;
        else if (newpos[0] >= L)
            newpos[0] = 0;

        if (newpos[1] < 0)
            newpos[1] = L - 1;
        else if (newpos[1] >= L)
            newpos[1] = 0;

        bool accept = false;

        if (INTERACTION_WALK == true)
        {
            int j1 = spin[pos[0]][pos[1]];
            int j2 = spin[newpos[0]][newpos[1]];
            
            double II = I[counter][j1] - I[counter][j2];

            if ((II < 0) || (casual() < exp(-II * F)))
                accept = true;
        }
        else
        {
            // we go from K to 0, where K > 0
            if (spin[newpos[0]][newpos[1]] == 0 && spin[pos[0]][pos[1]] > 0)
            // if (spin[newpos[0]][newpos[1]] > 0 && spin[newpos[0]][newpos[1]] < counter + 1)
            // if (spin[newpos[0]][newpos[1]] > 0 && spin[newpos[0]][newpos[1]] != counter + 1)
            {
                // decide whether to accept step
                if (casual() < 1. / exp(F))
                    accept = true;
            }
        }

        if (accept)
        {
            pos[0] = newpos[0];
            pos[1] = newpos[1];
        }

        tempo++;

    } while (counter < q); // q is the chain length
    return tempo;
}

double dEnergyReact(
    const int s, 
    const int i, const int j,
    const int im, const int jm,
    const int ip, const int jp)
{
    return (J[s][spin[i][jm]]   // down interaction
          + J[s][spin[i][jp]]   // up
          + J[s][spin[im][j]]   // left
          + J[s][spin[ip][j]]); // right
}

int rand_p(const double x[5], const double z)
{
    // cumulative probability
    double P_cum = 0.0;

    // random number
    double cas = casual();

    for (int k = 0; k < 5; k++){
        double P = x[k] / z;
        P_cum += P;
        if (cas < P_cum)
        {
            return k;
        }
    }
    return -1;
}

int time_to_react_new(const double beta, const double interaction, const int react_counter)
{
    /* simulating a substrate entering and reacting,
    it returns the total time to go through the pathway,
    F is the attraction energy of the substrate with the enzymes (uniform) */
    int counter = 0;
    
    int pos[2];
    int newpos[2];

    int tempo = 0;

    double delta[5];
    double P[5];
    double X[5];

    int x[5];
    int y[5];

    pos[0] = int(casual() * L);
    pos[1] = int(casual() * L);

    do
    {
        // l_react[pos[0]][pos[1]][counter] += 1;

        if (spin[pos[0]][pos[1]] == counter + 1)
        {
            counter++;
        }

        x[0] = pos[0];
        y[0] = pos[1];

        x[1] = x[0];
        y[1] = (y[0] + 1) % L;
        
        x[2] = x[0];
        y[2] = (L + y[0] - 1) % L;

        x[3] = (x[0] + 1) % L;
        y[3] = y[0];

        x[4] = (L + x[0] - 1) % L;
        y[4] = y[0];

        int s0 = spin[x[0]][y[0]];
        int s1 = spin[x[1]][y[1]];
        int s2 = spin[x[2]][y[2]];
        int s3 = spin[x[3]][y[3]];
        int s4 = spin[x[4]][y[4]];

        double F[5];
        F[0] = I[counter][s0];
        F[1] = I[counter][s1];
        F[2] = I[counter][s2];
        F[3] = I[counter][s3];
        F[4] = I[counter][s4];

        // z ... partition sum
        double z = 0.0;

        for (int k = 0; k < 5; k++){

            delta[k] = -F[k];

            X[k] = exp(-beta * delta[k]);
            z += X[k];
        }
        
        int k = rand_p(X, z);

        pos[0] = x[k];
        pos[1] = y[k];

        tempo++;

    } while (counter < q); // q is the chain length
    return tempo;
}

void load_lattice(const double beta)
{
    char fname[64];
    sprintf(fname, "lattice_%.8f.csv", beta);

    ifstream infile(fname);

    for (int i = 0; i < L; i++)
    {
        for (int j = 0; j < L; j++)
        {
            infile >> spin[i][j];
        }
    }
}

void save_lattice(const int t_index)
{
    char fname[64];
    
    bool txt_out = false;
    
    if (txt_out) // output as text
    {
        sprintf(fname, "lattice_%d.csv", t_index);

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
    else // binary
    {
        sprintf(fname, "lattice_%d.bin", t_index);

        ofstream fout;
        fout.open(fname, std::ios_base::out | std::ios_base::binary);

        for (int i = 0; i < L; i++)
        {
            fout.write(reinterpret_cast<const char*>(spin[i]), L * sizeof(spin[i][0]));
        }
        fout.close();
    }
}

void save_l_react(const int t)
{
    for (int k = 0; k < q + 1; k++)
    {
        char fname[64];
        sprintf(fname, "l_react_%d_%d.csv", t, k);

        ofstream fout;
        fout.open(fname);

        for (int i = 0; i < L; i++)
        {
            for (int j = 0; j < L; j++)
            {
                fout << l_react[i][j][k] << " ";
            }
            fout << endl;
        }
        fout.close();
    }
}


void load_sweep_vars(
    vector<double> &alpha,
    vector<double> &beta,
    vector<double> &mu)
{
    double v;
    
    ifstream f0("sweep_alpha.csv");
    if (f0.is_open()) {
        while (f0.good()) {
            f0 >> v;
            alpha.push_back(v);
        }
    }

    ifstream f1("sweep_beta.csv");
    if (f1.is_open()) {
        while (f1.good()) {
            f1 >> v;
            beta.push_back(v);
        }
    }

    ifstream f2("sweep_mu.csv");
    if (f2.is_open()) {
        while (f2.good()) {
            f2 >> v;
            mu.push_back(v);
        }
    }
}

void load_range_vars(
    vector<double> &interaction)
{
    double v;
    
    ifstream f0("range_interaction.csv");
    if (f0.is_open()) {
        while (f0.good()) {
            f0 >> v;
            interaction.push_back(v);
        }
    }
}

void load_interaction()
{
    ifstream infile_I("I.csv");
    ifstream infile_J("J.csv");

    for (int i = 0; i < q + 1; i++)
    {
        for (int j = 0; j < q + 1; j++)
        {
            infile_I >> I[i][j];
            infile_J >> J[i][j];
        }
    }
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

    for (int i = 0; i < q + 1; i++)
    {
        for (int j = 0; j < q + 1; j++)
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

void serialize_lattice_array(
    const int arr[][L][L],
    const int size,
    char *fbase)
{
    for (int k = 0; k < size; k++)
    {
        ofstream fout;

        char fname[64];
        sprintf(fname, "%s_%d", fbase, k);

        fout.open(fname, ios_base::app);

        for (int i = 0; i < L; i++)
        {
            for (int j = 0; j < L; j++)
            {
                fout << arr[k][i][j] << " ";
            }
            fout << endl;
        }
        fout.close();
    }
}

void reaction(
    vector<double> &beta,
    vector<double> &mu,
    const int sim_react)
{
    vector<int> tempo(sim_react);

    char fname[64];

    for (int j = 0; j < beta.size(); j++)
    {
        double b = beta[j];

        load_lattice(j);

        for (int i = 0; i < L; i++)
            for (int j = 0; j < L; j++)
                for (int k = 0; k < q; k++)
                    l_react[i][j][k] = 0;

        for (int i = 0; i < sim_react; i++)
        {
            // simulate the pathway
            // tempo[i] = time_to_react(interaction * b, interaction, i);
        }

        sprintf(fname, "reaction_tempo.csv");
        save_vector(tempo, fname);

        if (LOG_CONCENTRATION == true)
            save_l_react(b);
    }
}

void condensation(
    vector<double> &alpha,
    vector<double> &beta,
    vector<double> &mu,
    vector<double> &interaction,
    const int sim_cond,
    const int sim_react)
{
    vector<double> cond_energy(sim_cond);

    char fname[64];
    
    assert (beta.size() == mu.size());
    assert (beta.size() == alpha.size());

    vector<int> tempo(sim_react);

    for (int it = 0; it < beta.size(); it++)
    {
        double a = alpha[it];
        double b = beta[it];
        double m = mu[it];

        for (int i = 0; i < sim_cond; i++)
        {
            // simulate the condensation            
            cond_energy[i] = sweep(b, m, a);
        }
        
        for (int j = 0; j < interaction.size(); j++)
        {
            for (int i = 0; i < sim_react; i++)
            {
                // simulate the pathway
                tempo[i] = time_to_react(interaction[j] * b, interaction[j], i);
            }

            sprintf(fname, "reaction_tempo_%d.csv", j);
            save_vector(tempo, fname);
        }

        //save_substrate(it);
        save_lattice(it);

        if (DO_REACTION)
        {
            sprintf(fname, "substrate_%d", it);
            serialize_lattice_array(substrate, q + 1, fname);
        }

        sprintf(fname, "cond_energy.csv");
        save_vector(cond_energy, fname);
    }
}

int core(
    const int sim_cond,
    const int sim_react)
{
    srand(time(0));

    init_lattice(volume_frac);
    init_diffusion_kernel();

    load_interaction();

    vector<double> alpha;
    vector<double> beta;
    vector<double> mu;
    
    load_sweep_vars(alpha, beta, mu);

    vector<double> interaction;
    load_range_vars(interaction);

    if (sim_cond > 0)
    {
        condensation(alpha, beta, mu, interaction, sim_cond, sim_react);
    }

    // OUTDATED FOR NOW
    // if (sim_react > 0)
    // {
    //     reaction(beta, mu, sim_react);
    // }

    return 0;
}

int main(int argc, char **argv)
{
    int i = 0;

    // i++;
    // interaction = 1;
    // if (argc > i)
    //     interaction = atof(argv[i]);

    i++;
    volume_frac = 0.3;
    if (argc > i)
        volume_frac = atof(argv[i]);

    i++;
    int sim_cond = 1000;
    if (argc > i)
        sim_cond = atoi(argv[i]);

    i++;
    int sim_react = 1000;
    if (argc > i)
        sim_react = atoi(argv[i]);

    return core(sim_cond, sim_react);
}