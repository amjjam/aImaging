/******************************************************************************
 * This is class RCAMERA. It is a child of class CAMERA and implements a      *
 * rectangular image plane with pixels distributed at even internals in theta *
 * and phi directions.                                                        *
 ******************************************************************************/

#include "../include/rcamera.H"

#include <iostream>

/*=============================================================================
  RCAMERA(double T, double P, int nX, int nY, MPI_Comm comm) -
  constructor which uses duplicate communicator.

  double T, P - camera FOV in theta (vertical) and phi (horizontal) direction.
  int nX, nY - camera number of pixels in horizontal (nX) and veritcal (nY) 
  direction
  ============================================================================*/
RCAMERA::RCAMERA(double T, double P, int n, int m, MPI_Comm comm):
  CAMERA(comm,true){
  initialize(T,P,nX,nY);
}


/*=============================================================================
  RCAMERA(MPI_Comm comm, double T, double P, int nX, int nY) -
  constructor which uses the communicator provided
  ============================================================================*/
RCAMERA::RCAMERA(MPI_Comm comm, double T, double P, int nX, int nY):
  CAMERA(comm,false){
  initialize(T,P,nX,nY);
}


/*=============================================================================
  ~RCAMERA() - destructor
  ============================================================================*/
RCAMERA::~RCAMERA(){

}


/*=============================================================================
  RIMAGE *snap(LOS &m) - take an image of the model using the
  current camera position and orientation and return a pointer to the
  image. The caller will need to deallocate the image.

  The image communicator will be set in accordance with how the camera
  communicator was set. If the camera communicator was a duplicate
  then the image communicator will also be a duplicate. If the camera
  communicator was not a duplicate then the image communicator will
  not be a duplicate either.
  ============================================================================*/
RIMAGE *RCAMERA::snap(LOS &model){
  RIMAGE *d;
  if(duplicateCommunicator)
    d=new RIMAGE(nX,nY,comm,root);
  else 
    d=new RIMAGE(comm,nX,nY,root);
  
  int nIndex=d->nIndex();
  int i,j;
  for(int k=0;k<nIndex;k++){
    d->indexToPixel(k,i,j);
    d->setValue(k,model.los(pos,direction(i,j)));
  }  

  d->setPos(pos);
  d->setDir(roll,pitch,yaw);

  return d;
}


/*=============================================================================
  aVec direction(double i, double j, int c=1) - get the look-direction
  vector for pixel (i,j)
  
  double i,j is the pixel number which is integer, but a real number can also
  be passed to return the look direction of a point within a pixel.

  int c is the coordinate system. The default is 1 which is the
  external coordinate system. 0 is the internal coordinate system of
  the camera.
  ============================================================================*/
aVec RCAMERA::direction(double i, double j, int c){
  double t,p;
  
  t=M_PI/2-T/2+dt/2+i*dt;
  p=-P/2+dp/2+j*dp;

  aVec v=unitVector(t,p);

  if(c==0)
    return v;

  return transform(v);
}


/*=============================================================================
  void initialize(double T, double P, int nX, int nY) - initialization
  ============================================================================*/
void RCAMERA::initialize(double T, double P, int nX, int nY){
  RCAMERA::T=T;
  RCAMERA::P=P;
  RCAMERA::nX=nX;
  RCAMERA::nY=nY;
  dt=T/nY;
  dp=P/nX;
}
