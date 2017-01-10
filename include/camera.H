/******************************************************************************
 * This is base class CAMERA. It is the base class for any camera class       *
 ******************************************************************************/


#ifndef _CAMERA_H_
#define _CAMERA_H_

#include <aVec.H>
#include <aTime.H>
#include "model.H"
#include "image.H"

class CAMERA{
public:
  CAMERA(MPI_Comm comm, bool dup=false,int root=0);
  ~CAMERA();
  void setDir(double roll, double pitch, double yaw);
  void setDir(aVec dir, double roll=0);
  void setPos(aVec p);
  virtual IMAGE *snap(MODEL &m)=0;
  aVec unitVector(double t, double p);
protected:
  aVec pos;
  aVec dir;
  double roll,pitch,yaw;
  MPI_Comm comm;
  bool duplicateCommunicator;
  int root;

  aVec transform(aVec &v);
};

#endif
