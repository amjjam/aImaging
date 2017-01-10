/******************************************************************************
 * This is base class CAMERA. It is the base class for any camera class       *
 ******************************************************************************/

#include "../include/camera.H"

/*=============================================================================
  CAMERA(MPI_Comm comm, bool dup=false,root=0) - constructor. Sets the
  communicator for the camera. If dup=true then use a duplicate
  communicator, otherwise use the communicator provided.

  MPI_Comm comm - communicator
  dup - duplication
  root - the rank to use as root. 
  ============================================================================*/
CAMERA::CAMERA(MPI_Comm comm, bool dup, int root){
  if(!dup){
    CAMERA::comm=comm;
    duplicateCommunicator=false;
  }  
  CAMERA::root=root;
}


/*=============================================================================
  ~CAMERA() - destructor
  ============================================================================*/
CAMERA::~CAMERA(){

}


/*=============================================================================
  void setDir(double roll, double pitch, double yaw) - set the
  orientation of the camera using roll, pitch, and yaw angles in radians
  ============================================================================*/
void CAMERA::setDir(double roll, double pitch, double yaw){
  CAMERA::roll=roll;
  CAMERA::pitch=pitch;
  CAMERA::yaw=yaw;
}


/*=============================================================================
  void setDir(aVec dir, double roll) - set the orientation of the camera
  with a roll angle and direction vector
  ============================================================================*/
void CAMERA::setDir(aVec dir, double roll){
  pitch=-asin(dir.Z()/length(dir));
  yaw=atan2(dir.Y(),dir.X());
  CAMERA::roll=roll;
}


/*=============================================================================
  void setPos(aVec p) - set the position of the camera
  ============================================================================*/
void CAMERA::setPos(aVec p){
  pos=p;
}


/*=============================================================================
  aVec unitVector(double t, double p) - return a unit vector pointing
  in the direction (t,p)
  ============================================================================*/
aVec CAMERA::unitVector(double t, double p){
  return aVec(sin(t)*cos(p),sin(t)*sin(p),cos(t));
}


/*=============================================================================
  aVec transform(aVec &v) - transforms a vector from the camera
  coordinate system to the model coordinate system.
  ============================================================================*/
aVec CAMERA::transform(aVec &v){
 
  aVec r;

  // Rotate by roll angle
  r=v.rotateX(roll);
  
  // Rotate by the pitch angle
  r=r.rotateY(pitch);

  // Rotate by the yaw angle
  return r.rotateZ(yaw);
}
