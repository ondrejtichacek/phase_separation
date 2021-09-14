#include "iostream"
#include "stdlib.h"
#include "math.h"
#include "time.h"
#include "fstream"
#include "vector"
#include "string"

using namespace std;

double casual(){                  // a random number uniform in (0,1)
    long double   x = double  (random())/(RAND_MAX+1.0);
    return x;
}

const int L = 100;                            // size of the system: 2D square lattice  LxL periodic boundary conditions
const int q = 20;                             // number of enzymes (length of the pathway)
double volume_frac=0.3;                  // volume fraction of the solutes
double interaction=1;                     // interaction between substrate and enzymes (uniform)
double J[q+1][q+1];                           //  interaction matrix    
int spin[L][L];                               // lattice variables



void init(double conc){    // initialization: filling interaction matrix & the lattice variables (uniform volume fraction) 

          for(int i=0;i<=q;i++) J[0][i]=J[i][0]=0;                        // interaction with solute (=0 default)
          for(int i=1;i<=q;i++){
              for(int j=i;j<=q;j++){
                         //J[i][j]=  2*(0.5-random(1));
                         if(i==j) J[i][j]=1;                     // self interaction  (=1 default)
                         if(j>=i+1) J[i][j]=1;                   // cross interactions (=1 default)
                         J[j][i]=J[i][j];                       
                         }
               }
          for(int i=0;i<=L-1;i++){     // filling the lattice with volume fraction "conc" at random (beta=0)
               for(int j=0;j<=L-1;j++){
                         double cas = casual();
                         spin[i][j]=0;                     
                         for(int k=1;k<=q;k++) if(cas>=(k-1)*conc/q && cas< k*conc/q) spin[i][j]=k;
                                   }
                         }
}




void kawasaki(double beta){         // Kawasaki Montecarlo:  exchange two particles position
                  int i1 =int(casual()*L);
                  int j1 =int(casual()*L);
                  int ip = (i1+1)%L;
                  int im = (L+i1-1)%L;
                  int jp = (j1+1)%L;
                  int jm = (L+j1-1)%L;  
                  int i2 =int(casual()*L);
                  int j2 =int(casual()*L);
                  int ip2 = (i2+1)%L;
                  int im2 = (L+i2-1)%L;
                  int jp2 = (j2+1)%L;
                  int jm2 = (L+j2-1)%L; 
                  if(spin[i1][j1]!=spin[i2][j2]){                    
                      double F1 = J[ spin[i1][j1] ][ spin[i1][jm] ] +  J[ spin[i1][j1] ][ spin[i1][jp] ]  
   +  J[ spin[i1][j1] ][ spin[im][j1] ] +  J[ spin[i1][j1] ][ spin[ip][j1] ] ;                                 
                      double F1n =  J[ spin[i1][j1] ][ spin[i2][jm2] ] +  J[ spin[i1][j1] ][ spin[i2][jp2] ]  
   +  J[ spin[i1][j1] ][ spin[im2][j2] ] +  J[ spin[i1][j1] ][ spin[ip2][j2] ] ;    
                      double F2n = J[ spin[i2][j2] ][ spin[i1][jm] ] +  J[ spin[i2][j2] ][ spin[i1][jp] ]  
   +  J[ spin[i2][j2] ][ spin[im][j1] ] +  J[ spin[i2][j2] ][ spin[ip][j1] ] ;                                
                      double F2 =  J[ spin[i2][j2] ][ spin[i2][jm2] ] +  J[ spin[i2][j2] ][ spin[i2][jp2] ]  
   +  J[ spin[i2][j2] ][ spin[im2][j2] ] +  J[ spin[i2][j2] ][ spin[ip2][j2] ] ;  
                      double delta = F1 + F2 - F1n - F2n;              // energy variation for the swap
                      double cas = casual();                                        
                      int news=-1;
                      if(delta<=0 || cas<=1/exp(beta*delta)) news=1;           // Metropolis rule
                      if(news==1){
                       int sp =spin[i1][j1];
                       int sp2 =spin[i2][j2];
                       spin[i2][j2]=sp;
                       spin[i1][j1]=sp2;    
                      }
                }                    
}


void sweep(double beta){                                                          // sweep over all system
                         for(int i=0;i<=L*L-1;i++) kawasaki(beta);

}




double time_to_react( double F ){   
                                        // simulating a substrate entering and reacting, it returns the total time to go through the pathway, F is the attraction energy of the substrate with the enzymes (uniform) 
                            int counter=0;
                            int pos[2];
                            double tempo=0;
                            pos[0] =int(casual()*L);
                            pos[1]=int(casual()*L);
                            do{
                                if(spin[pos[0]][pos[1]]==counter+1) counter++;
                                double cas =casual();
                                int newpos[2];
                                if(cas<0.5) newpos[0] = pos[0]+1;
                                else newpos[0]=pos[0]-1;
                                if(newpos[0]<0) newpos[0]=L-1;
                                if(newpos[0]>=L) newpos[0]=0;
                                cas = casual();
                                if(cas<0.5) newpos[1] = pos[1]+1;
                                else newpos[1]=pos[1]-1;
                                if(newpos[1]<0) newpos[1]=L-1;
                                if(newpos[1]>=L) newpos[1]=0;
                                int okkei=1;
                                if(spin[ newpos[0] ][newpos[1]]==0 && spin[pos[0]][pos[1]]>0 ){
                                           cas = casual();
                                           okkei=0;
                                           if(cas<1./exp(F)) okkei=1;
                                          }
                                          
                                if(okkei==1){
                                           pos[0]=newpos[0];
                                           pos[1]=newpos[1];
                                          }
                                tempo++;
                               }while(counter<q);
                              return tempo; 
}






int main(){
                   srand(time(0));
                   
                   init(volume_frac);
                   double betamin=0;
                   double betamax=10;
                   int sim_cond=1000;
                   int sim_react=1000;
                   int number = 100;
                   for(int j=1;j<=number;j++){
                        double beta = betamin+(betamax-betamin)*double(j)/double(number);// inverse temperature (cooling) 
                        for(int i=0;i<=sim_cond;i++) sweep(beta);         // simulate the condensation
                            double tempo=0;
                            for(int k=0;k<=sim_react;k++) tempo += time_to_react(interaction*beta);   // simulate the pathway 
                            cout << beta << "  " << tempo/double(sim_react) << endl; 
                                  // output: inverse temp vs average time to react
                            }                      
                    
                                         
    
    
     return 0 ;
}
