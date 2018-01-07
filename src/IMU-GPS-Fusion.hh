#include "smctc.hh"
#include <armadillo>
#include "../Coordinate-Transforms/src/CoordinateTransform.h"
#include "../BNO055-Linux-Library/src/BNO055.h"

class cv_state
{
public:
  cv_state();
  arma::vec stateSpace(15); //Theta, Omega, Pos, Velocity, Acc, each contains x, y, z
  
};

class GPS_obs
{
public:
  bool newData= FALSE;
  GPS_obs();
  arma::vec measurementGPS(6); //measure xyz pos and xyz speed
  arma::mat CovarianceMatrixR(6, 6);
  void SetMeasurement (arma::vec Pos, arma::vec Velocity, double Xerror, double Yerror, double Zerror, double Verror, double Cerror);

};

class IMU_obs
{
public:
   bool newData = FALSE;
   IMU_obs();
   IMU_obs(double accError, double gyroError, double poseError);
   arma::vec measurementIMU(9); //measure xyz acc, xyz angular speed, xyz pose
   arma::mat CovarianceMatrixR(9, 9);
   void SetMeasurement (arma::vec Theta,  arma::vec Omega, arma::vec Acc);
  
};

double logLikelihood(long lTime, const cv_state & X);

smc::particle<cv_state> fInitialise(smc::rng *pRng);

long fSelect(long lTime, const smc::particle<cv_state> & p, 
	     smc::rng *pRng);
void GPSKernel(long lTime, smc::particle<cv_state> & pFrom, 
	   smc::rng *pRng);//movements for GPS measurements
void IMUKernel(long lTime, smc::particle<cv_state> & pFrom, 
	   smc::rng *pRng); //movement for IMU measurements


extern GPS_obs * y_gps; 
extern IMU_obs * y_imu; 
