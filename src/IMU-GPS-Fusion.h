#include "../include/smctc.h"
#include <armadillo>
#include "../Coordinate-Transforms/src/CoordinateTransform.h"
#include "../BNO055-Linux-Library/src/BNO055.h"

#define N 4096

using namespace arma;
using namespace std;

class cv_state
{
public:
  cv_state();
  arma::vec stateSpace;//(15); //Theta, Omega, Pos, Velocity, Acc, each contains x, y, z
  
};

class GPS_obs
{
public:
  GPS_obs(double time);
  arma::vec measurementGPS;//(6); //measure xyz pos and xyz speed
  arma::mat CovarianceMatrixR;//(6, 6);
  arma::mat sensorH;//(3, 15);
  void SetMeasurement (arma::vec Pos, arma::vec Velocity, double Xerror, double Yerror, double Zerror, double Verror, double Cerror, double currentT);
  bool newData;
   double deltaT;
   double currentTime;
   double lastTime;

};

class IMU_obs
{
public:
   IMU_obs();
   IMU_obs(double accError, double gyroError, double poseError, double time);
   arma::vec measurementIMU;//(9); //measure xyz acc, xyz angular speed, xyz pose
   arma::mat CovarianceMatrixR;//(9, 9);
   arma::mat sensorH;//(3, 15);
   void SetMeasurement (arma::vec Theta,  arma::vec Omega, arma::vec Acc, double currentT);
   bool newData;
   double deltaT;
   double currentTime;
   double lastTime;
  
};

double logLikelihoodIMU(const cv_state & X, const IMU_obs & y_imui);

double logLikelihoodGPS(const cv_state & X, const GPS_obs & y_gpsi);


smc::particle<cv_state> fInitialise(smc::rng *pRng);

long fSelect(long lTime, const smc::particle<cv_state> & p, 
	     smc::rng *pRng);
void GPSKernel(long lTime, smc::particle<cv_state> & pFrom, 
	   smc::rng *pRng);//movements for GPS measurements
void IMUKernel(long lTime, smc::particle<cv_state> & pFrom, 
	   smc::rng *pRng); //movement for IMU measurements


extern GPS_obs * y_gps; 
extern IMU_obs * y_imu; 
