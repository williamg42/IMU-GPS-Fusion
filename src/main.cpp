#include "smctc.hh"
#include "IMU-GPS-Fusion.hh"
#include <cstdio> 
#include <cstdlib>
#include <cstring>

using namespace std;

///The observations

int main(int argc, char** argv)
{
  
  try {
void (*pfMoves[])(long, smc::particle<mChain<double> > &,smc::rng*) = {GPSKernel, IMUKernel};
smc::moveset<cv_state> Moveset(fInitialise, fSelect,sizeof(pfMoves) /sizeof(pfMoves[0]),pfMoves, fMCMC);
smc::sampler<cv_state> Sampler(N, SMC_HISTORY_NONE);  
Sampler.SetResampleParams(SMC_RESAMPLE_RESIDUAL,0.5);
Sampler.SetMoveSet(Moveset);
Sampler.Initialise();

    
    for(int n=1 ; n < N ; ++n) {
      Sampler.Iterate();
      
      double xm,xv,ym,yv;
      xm = Sampler.Integrate(integrand_mean_x,NULL);
      xv = Sampler.Integrate(integrand_var_x, (void*)&xm);
      ym = Sampler.Integrate(integrand_mean_y,NULL);
      yv = Sampler.Integrate(integrand_var_y, (void*)&ym);
      
      cout << xm << "," << ym << "," << xv << "," << yv << endl;
    }
  }

  catch(smc::exception  e)
    {
      cerr << e;
      exit(e.lCode);
    }
}



double integrand_mean_x(const cv_state& s, void *)
{
  return s.x_pos;
}

double integrand_var_x(const cv_state& s, void* vmx)
{
  double* dmx = (double*)vmx;
  double d = (s.x_pos - (*dmx));
  return d*d;
}

double integrand_mean_y(const cv_state& s, void *)
{
  return s.y_pos;
}

double integrand_var_y(const cv_state& s, void* vmy)
{
  double* dmy = (double*)vmy;
  double d = (s.y_pos - (*dmy));
  return d*d;
}
