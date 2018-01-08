#include <iostream>
#include <cmath>
#include <gsl/gsl_randist.h>

#include "smctc.hh"
#include "IMU-GPS-Fusion.hh"

using namespace std;

  cv_state::cv_state() {
    
  }

  GPS_obs::GPS_obs() {
    CovarianceMatrixR.fill(0.0);
    measurementGPS.fill(0.0);
    sensorH.fill(0.0);
    sensorH(6,0) = 1;
    sensorH(7,1) = 1;
    sensorH(8,2) = 1;
    sensorH(9,0) = 1;
    sensorH(10,1) = 1;
    sensorH(11,2) = 1;
  }

  void GPS_obs::SetMeasurement (arma::vec Pos, arma::vec Velocity, double Xerror, double Yerror, double Zerror, double Verror, double Cerror, double deltaTime) {
    CovarianceMatrixR(0,0) = Xerror*Xerror;
    CovarianceMatrixR(1,1) = Yerror*Yerror;
    CovarianceMatrixR(2,2) = Zerror*Zerror;
    CovarianceMatrixR(3,3) = Verror*Verror;
    CovarianceMatrixR(4,4) = Verror*Verror;
    CovarianceMatrixR(5,5) = Cerror*Cerror;
    deltaT = deltaTime;
    measurementGPS.span(0,2) = Pos;
    measurementGPS.span(3,5) = Velocity;
    
      
      
  }


   IMU_obs::IMU_obs() {
     
    measurementIMU.fill(0.0);
    CovarianceMatrixR.fill(0.0); 
    sensorH.fill(0.0);
    sensorH(0,0) = 1;
    sensorH(1,1) = 1;
    sensorH(2,2) = 1;
    sensorH(3,0) = 1;
    sensorH(4,1) = 1;
    sensorH(5,2) = 1;
    sensorH(12,0) = 1;
    sensorH(13,1) = 1;
    sensorH(14,2) = 1;
     
   }
   IMU_obs::IMU_obs(double poseError, double gyroError, double accError) {
    measurementIMU.fill(0.0);
    CovarianceMatrixR.fill(0.0); 
    sensorH.fill(0.0);
    CovarianceMatrixR(0,0) = poseError*poseError;
    CovarianceMatrixR(1,1) = poseError*poseError;
    CovarianceMatrixR(2,2) = poseError*poseError;
    CovarianceMatrixR(3,3) = gyroError*gyroError;
    CovarianceMatrixR(4,4) = gyroError*gyroError;
    CovarianceMatrixR(5,5) = gyroError*gyroError;
    CovarianceMatrixR(6,6) = accError*accError;
    CovarianceMatrixR(7,7) = accError*accError;
    CovarianceMatrixR(8,8) = accError*accError;
     
   }

   void IMU_obs::SetMeasurement (arma::vec Theta,  arma::vec Omega, arma::vec Acc, double deltaTime) {
     
         
    measurementIMU.span(0,2) = Theta;
    measurementIMU.span(3,5) = Omega;
    measurementIMU.span(6,8) = Acc;
    deltaT = deltaTime;
     
   }
  


///The function corresponding to the log likelihood at specified time and position (up to normalisation)

///  \param lTime The current time (i.e. the index of the current distribution)
///  \param X     The state to consider 
double logLikelihoodIMU(const cv_state & X)
{
  arma::mat temp = y_imu.measurementIMU-y_imu.sensorH*X.stateSpace;
  arma::mat temp2 = arma::trans(temp);
  arma::mat Rinv = arma::inv(y_imu.CovarianceMatrixR);
  
  arma::mat weight = -0.5*temp2*Rinv;
  
  return (double) arma::cumprod(weight)(0);
}

double logLikelihoodGPS(const cv_state & X)
{
  arma::mat temp = y_gps.measurementGPS-y_gps.sensorH*X.stateSpace;
  arma::mat temp2 = arma::trans(temp);
  arma::mat Rinv = arma::inv(y_gps.CovarianceMatrixR);
  
  arma::mat weight = -0.5*temp2*Rinv;
  
  return (double) arma::cumprod(weight)(0);
}


///A function to initialise particles

/// \param pRng A pointer to the random number generator which is to be used
smc::particle<cv_state> fInitialise(smc::rng *pRng)
{
  cv_state value;
  
  value(0) = pRng->Normal(0,sqrt(var_s0));
  value(0) = pRng->Normal(0,sqrt(var_s0));
  value(0) = pRng->Normal(0,sqrt(var_u0));
  value(0) = pRng->Normal(0,sqrt(var_u0));

  return smc::particle<cv_state>(value,logLikelihood(0,value));
}

void GPSKernel(long lTime, smc::particle<cv_state> & pFrom, smc::rng *pRng) {//movements for GPS measurements
  cv_state * k = pFrom.GetValuePointer();
  
  k.stateSpace(0) = k.stateSpace(0)+y_gps.deltaT*k.stateSpace(3)+ pRng->Normal(0,sqrt(var_s0)); //x pose
  k.stateSpace(1) = k.stateSpace(1)+y_gps.deltaT*k.stateSpace(4)+ pRng->Normal(0,sqrt(var_s0));//y pose
  k.stateSpace(2) = k.stateSpace(2)+y_gps.deltaT*k.stateSpace(5)+ pRng->Normal(0,sqrt(var_s0));//z pose
  k.stateSpace(3) = k.stateSpace(3)+pRng->Normal(0,sqrt(var_s0)); //x angular velocity
  k.stateSpace(4) = k.stateSpace(4)+pRng->Normal(0,sqrt(var_s0)); //y angular velocity
  k.stateSpace(5) = k.stateSpace(5)+pRng->Normal(0,sqrt(var_s0)); //z angular velocity
  k.stateSpace(6) = k.stateSpace(6)+y_gps.deltaT*k.stateSpace(9)+y_gps.deltaT*y_gps.deltaT*0.5*k.stateSpace(12)+pRng->Normal(0,sqrt(var_s0)); //x position
  k.stateSpace(7) = k.stateSpace(7)+y_gps.deltaT*k.stateSpace(10)+y_gps.deltaT*y_gps.deltaT*0.5*k.stateSpace(13)+pRng->Normal(0,sqrt(var_s0)); //y position
  k.stateSpace(8) = k.stateSpace(8)+y_gps.deltaT*k.stateSpace(11)+y_gps.deltaT*y_gps.deltaT*0.5*k.stateSpace(14)+pRng->Normal(0,sqrt(var_s0)); //z position
  k.stateSpace(9) = k.stateSpace(9)+y_gps.deltaT*k.stateSpace(12)+pRng->Normal(0,sqrt(var_s0)); //x velocity
  k.stateSpace(10) = k.stateSpace(10)+y_gps.deltaT*k.stateSpace(13)+pRng->Normal(0,sqrt(var_s0)); //x velocity
  k.stateSpace(11) = k.stateSpace(11)+y_gps.deltaT*k.stateSpace(14)+pRng->Normal(0,sqrt(var_s0)); //x velocity
  k.stateSpace(12) = k.stateSpace(12)+pRng->Normal(0,sqrt(var_s0)); //x angular velocity
  k.stateSpace(13) = k.stateSpace(13)+pRng->Normal(0,sqrt(var_s0)); //y angular velocity
  k.stateSpace(14) = k.stateSpace(14)+pRng->Normal(0,sqrt(var_s0)); //z angular velocity
  
  pFrom.MultiplyLogWeightBy(logLikelihoodGPS(*k));
  
}
void IMUKernel(long lTime, smc::particle<cv_state> & pFrom, smc::rng *pRng) {//movement for IMU measurements
  cv_state * k = pFrom.GetValuePointer();
  
  k.stateSpace(0) = k.stateSpace(0)+y_gps.deltaT*k.stateSpace(3)+ pRng->Normal(0,sqrt(var_s0)); //x pose
  k.stateSpace(1) = k.stateSpace(1)+y_gps.deltaT*k.stateSpace(4)+ pRng->Normal(0,sqrt(var_s0));//y pose
  k.stateSpace(2) = k.stateSpace(2)+y_gps.deltaT*k.stateSpace(5)+ pRng->Normal(0,sqrt(var_s0));//z pose
  k.stateSpace(3) = k.stateSpace(3)+pRng->Normal(0,sqrt(var_s0)); //x angular velocity
  k.stateSpace(4) = k.stateSpace(4)+pRng->Normal(0,sqrt(var_s0)); //y angular velocity
  k.stateSpace(5) = k.stateSpace(5)+pRng->Normal(0,sqrt(var_s0)); //z angular velocity
  k.stateSpace(6) = k.stateSpace(6)+y_gps.deltaT*k.stateSpace(9)+y_gps.deltaT*y_gps.deltaT*0.5*k.stateSpace(12)+pRng->Normal(0,sqrt(var_s0)); //x position
  k.stateSpace(7) = k.stateSpace(7)+y_gps.deltaT*k.stateSpace(10)+y_gps.deltaT*y_gps.deltaT*0.5*k.stateSpace(13)+pRng->Normal(0,sqrt(var_s0)); //y position
  k.stateSpace(8) = k.stateSpace(8)+y_gps.deltaT*k.stateSpace(11)+y_gps.deltaT*y_gps.deltaT*0.5*k.stateSpace(14)+pRng->Normal(0,sqrt(var_s0)); //z position
  k.stateSpace(9) = k.stateSpace(9)+y_gps.deltaT*k.stateSpace(12)+pRng->Normal(0,sqrt(var_s0)); //x velocity
  k.stateSpace(10) = k.stateSpace(10)+y_gps.deltaT*k.stateSpace(13)+pRng->Normal(0,sqrt(var_s0)); //x velocity
  k.stateSpace(11) = k.stateSpace(11)+y_gps.deltaT*k.stateSpace(14)+pRng->Normal(0,sqrt(var_s0)); //x velocity
  k.stateSpace(12) = k.stateSpace(12)+pRng->Normal(0,sqrt(var_s0)); //x angular velocity
  k.stateSpace(13) = k.stateSpace(13)+pRng->Normal(0,sqrt(var_s0)); //y angular velocity
  k.stateSpace(14) = k.stateSpace(14)+pRng->Normal(0,sqrt(var_s0)); //z angular velocity
  
  pFrom.MultiplyLogWeightBy(logLikelihoodIMU(*k));
}


long fSelect(long lTime, const smc::particle<cv_state> & p, smc::rng *pRng) {
  
  if  (y_gps.newData == TRUE) return 1;
  else  return 0;
}
