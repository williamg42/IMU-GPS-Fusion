#include "../include/smctc.h"
#include "./IMU-GPS-Fusion.h"
#include <cstdio> 
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <libgpsmm.h>
#include <math.h>
#include <time.h> 
#include <sys/timeb.h>  

using namespace std;


GPS_obs * y_gps; 
IMU_obs * y_imu; 

double integrand_mean_x(const cv_state&, void*);
double integrand_mean_y(const cv_state&, void*);
double integrand_var_x(const cv_state&, void*);
double integrand_var_y(const cv_state&, void*);

int main(int argc, char** argv)
{

 struct timeb timer_msec;
  long long int timestamp_msec; /* timestamp in millisecond. */
  if (!ftime(&timer_msec)) {
    timestamp_msec = ((long long int) timer_msec.time) * 1000ll + 
                        (long long int) timer_msec.millitm;
  }
  else {
    timestamp_msec = -1;
  }
  printf("%lld milliseconds since epoch\n", timestamp_msec);


y_gps = new GPS_obs(timestamp_msec/1000);
y_imu = new IMU_obs(3,5,7, (timestamp_msec/1000.0)); //set variance of sensors here



CoordinateTransform transform;
FrameCoordinates coordinates;


y_gps->newData = 0;
y_imu->newData = 0;

BNO055 bno = BNO055(-1, BNO055_ADDRESS_A, 1);

bno.begin(bno.OPERATION_MODE_NDOF_FMC_OFF);

usleep(1000000);

 int temp = bno.getTemp();
  std::cout << "Current Temperature: " << temp << " C" << std::endl;

  bno.setExtCrystalUse(true);

usleep(1000000);

//
imu::Quaternion quat = bno.getQuat();

Quaternions q;

q.W = quat.w();
q.X = quat.x();
q.Y = quat.y();
q.Z = quat.z();




imu::Vector<3> acc_temp = bno.getVector(BNO055::VECTOR_ACCELEROMETER);

FrameCoordinates acc_bframe;

acc_bframe.first = acc_temp.x(); 
acc_bframe.second = acc_temp.y(); 
acc_bframe.third = acc_temp.z(); 

FrameCoordinates acc_NEDframe = transform.Bframe_to_NED(q, acc_bframe);

imu::Vector<3> ang_temp = bno.getVector(BNO055::VECTOR_GYROSCOPE);

FrameCoordinates gyro_bframe;

gyro_bframe.first = ang_temp.x(); 
gyro_bframe.second = ang_temp.y(); 
gyro_bframe.third = ang_temp.z(); 

FrameCoordinates gyro_NEDframe = transform.Bframe_to_NED(q, gyro_bframe);

imu::Vector<3> pose_temp = bno.getVector(BNO055::VECTOR_EULER);

FrameCoordinates pose;

pose.first = pose_temp.x(); 
pose.second = pose_temp.y(); 
pose.third = pose_temp.z();

y_imu->newData = 1;
arma::vec Theta = { pose.first, pose.second, pose.third };
arma::vec Omega = { gyro_NEDframe.first, gyro_NEDframe.second, gyro_NEDframe.third };
arma::vec Acc = { acc_NEDframe.first, acc_NEDframe.second, acc_NEDframe.third-9.81 };

y_imu->SetMeasurement (Theta,  Omega, Acc, (timestamp_msec/1000.0));

//



//

  gpsmm gps_rec("localhost", DEFAULT_GPSD_PORT);

  if (gps_rec.stream(WATCH_ENABLE | WATCH_JSON) == NULL) {
    std::cerr << "No GPSD running.\n";
    return 1;
  }

  
    struct gps_data_t *gpsd_data;

    if (!gps_rec.waiting(1000000)) {
      std::cerr << "GPSD timeout error.\n";
      return 2;
    }

    if ((gpsd_data = gps_rec.read()) == NULL) {
      std::cerr << "GPSD read error.\n";
      return 1;
    } else {
      while (((gpsd_data = gps_rec.read()) == NULL) ||
             (gpsd_data->fix.mode < MODE_2D)) {
        std::cout << "waiting for GPS fix" << std::endl;
        usleep(1000000);
      }
      timestamp_t ts   = gpsd_data->fix.time;
      coordinates.second  = gpsd_data->fix.latitude;
      coordinates.first = gpsd_data->fix.longitude;
	  coordinates.third = gpsd_data->fix.altitude;
      double heading = gpsd_data ->fix.track;
	  double speed = gpsd_data ->fix.speed;
      double climb = gpsd_data ->fix.climb;
  
      double xerror = gpsd_data->fix.epx;
      double yerror = gpsd_data->fix.epy;
      double zerror = gpsd_data ->fix.epv;
      double serror = gpsd_data ->fix.eps;
      double cerror = gpsd_data ->fix.epc;
      

      if (isnan(xerror))
		xerror = 1000;
      if (isnan(yerror))
		yerror = 1000;
      if (isnan(zerror))
		zerror = 1000;
      if (isnan(serror))
		serror = 1000;
      if (isnan(cerror))
		cerror = 1000;
      if (isnan(speed))
		speed = 0;
      if (isnan(climb))
		climb = 0;
if (isnan(heading))
		heading = 45;



transform.set_initialGeodetic(coordinates);

FrameCoordinates NED = transform.Geodetic_to_NED(coordinates);
  if (isnan(NED.first))
		NED.first = 0;
      if (isnan(NED.second))
		NED.second = 0;
      if (isnan(NED.third))
		NED.third = 0;
y_gps->newData = 1;
arma::vec Pos = { NED.first, NED.second, NED.third };
arma::vec Velocity = { cos(heading*2*M_PI)*speed, sin(heading*2*M_PI)*speed, climb };

y_gps->SetMeasurement(Pos, Velocity, xerror, yerror, zerror, serror, cerror, (timestamp_msec/1000));
std::cout << "New data" << std::endl;
std::cout << Pos << std::endl;
std::cout << Velocity << std::endl;


    }
  


//

  
  try {

void (*pfMoves[])(long, smc::particle<cv_state> &,smc::rng*) = {IMUKernel, GPSKernel};
smc::moveset<cv_state> Moveset(fInitialise, fSelect,sizeof(pfMoves) /sizeof(pfMoves[0]),pfMoves, NULL);
smc::sampler<cv_state> Sampler(N, SMC_HISTORY_NONE);  
Sampler.SetResampleParams(SMC_RESAMPLE_RESIDUAL, .8);
Sampler.SetMoveSet(Moveset);
Sampler.Initialise();

    for(int n=0 ; n <100 ; ++n) {
    
    //cout << "about to iterate filter" << endl;
      Sampler.Iterate();

y_gps->newData = 0;
y_imu->newData = 0;
      
      double xm,xv,ym,yv;
      xm = Sampler.Integrate(integrand_mean_x,NULL);
     xv = Sampler.Integrate(integrand_var_x, (void*)&xm);
      ym = Sampler.Integrate(integrand_mean_y,NULL);
      yv = Sampler.Integrate(integrand_var_y, (void*)&ym);
      
      cout << xm << "," << ym << "," << sqrt(xv) << "," << sqrt(yv) << endl;

//usleep(100000);

 if (!ftime(&timer_msec)) {
    timestamp_msec = ((long long int) timer_msec.time) * 1000ll + 
                        (long long int) timer_msec.millitm;
  }
  else {
    timestamp_msec = -1;
  }
 // printf("%lld milliseconds since epoch\n", timestamp_msec);



//
imu::Quaternion quat = bno.getQuat();

Quaternions q;

q.W = quat.w();
q.X = quat.x();
q.Y = quat.y();
q.Z = quat.z();




imu::Vector<3> acc_temp = bno.getVector(BNO055::VECTOR_ACCELEROMETER);

FrameCoordinates acc_bframe;

acc_bframe.first = acc_temp.x(); 
acc_bframe.second = acc_temp.y(); 
acc_bframe.third = acc_temp.z(); 

FrameCoordinates acc_NEDframe = transform.Bframe_to_NED(q, acc_bframe);

imu::Vector<3> ang_temp = bno.getVector(BNO055::VECTOR_GYROSCOPE);

FrameCoordinates gyro_bframe;

gyro_bframe.first = ang_temp.x(); 
gyro_bframe.second = ang_temp.y(); 
gyro_bframe.third = ang_temp.z(); 

FrameCoordinates gyro_NEDframe = transform.Bframe_to_NED(q, gyro_bframe);

imu::Vector<3> pose_temp = bno.getVector(BNO055::VECTOR_EULER);

FrameCoordinates pose;

pose.first = pose_temp.x(); 
pose.second = pose_temp.y(); 
pose.third = pose_temp.z();

y_imu->newData = 1;
arma::vec Theta = { pose.first, pose.second, pose.third };
arma::vec Omega = { gyro_NEDframe.first, gyro_NEDframe.second, gyro_NEDframe.third };
arma::vec Acc = { acc_NEDframe.first, acc_NEDframe.second, acc_NEDframe.third-9.81 };

if (!ftime(&timer_msec)) {
    timestamp_msec = ((long long int) timer_msec.time) * 1000ll + 
                        (long long int) timer_msec.millitm;
  }
  else {
    timestamp_msec = -1;
  }
 // printf("%lld milliseconds since epoch\n", timestamp_msec);


y_imu->SetMeasurement (Theta,  Omega, Acc, timestamp_msec/1000.0);

if ((gpsd_data = gps_rec.read()) != NULL)
{
y_gps->newData = 1;
arma::vec Pos = { 0, 0, 0 };
arma::vec Velocity = { 0, 0, 0 };

if (!ftime(&timer_msec)) {
    timestamp_msec = ((long long int) timer_msec.time) * 1000ll + 
                        (long long int) timer_msec.millitm;
  }
  else {
    timestamp_msec = -1;
  }
 // printf("%lld milliseconds since epoch\n", timestamp_msec);


y_gps->SetMeasurement(Pos, Velocity, 5, 5, 5, 5, 5, (timestamp_msec/1000));
std::cout << "New data" << std::endl;
}


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
//std::cout << s.stateSpace << std::endl;
  return  s.stateSpace(6);
}

double integrand_var_x(const cv_state& s, void* vmx)
{
  double* dmx = (double*)vmx;
  double d = (s.stateSpace(6) - (*dmx));
  return d*d;
}

double integrand_mean_y(const cv_state& s, void *)
{
  return s.stateSpace(7);
}

double integrand_var_y(const cv_state& s, void* vmy)
{
  double* dmy = (double*)vmy;
  double d = (s.stateSpace(7) - (*dmy));
  return d*d;
}



