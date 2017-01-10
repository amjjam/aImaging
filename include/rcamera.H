/******************************************************************************
 * This is class RCAMERA. It is a child of class CAMERA and implements a      *
 * rectangular image plane with pixels distributed at even internals in theta *
 * and phi directions.                                                        *
 ******************************************************************************/

#ifndef _RCAMERA_H_
#define _RCAMERA_H_

#include "camera.H"
#include "rimage.H"
#include <mpi.h>

class RCAMERA: public CAMERA{
public:
  RCAMERA(double T, double P, int nX, int nY, MPI_Comm comm);
  RCAMERA(MPI_Comm comm, double T, double P, int nX, int nY);
  ~RCAMERA();
  virtual RIMAGE *snap(LOS &m);
  aVec direction(double i, double j, int c=1);
private:
  double T,P;
  int nX,nY;
  double dt,dp;
  
  void initialize(double T, double P, int nX, int nY);
};

#endif
