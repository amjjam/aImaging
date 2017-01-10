/******************************************************************************
 * This is class IMAGE. It contains an image output by a camera together with *
 * associated metadata.                                                       *
 ******************************************************************************/

#include "../include/image.H"
#include <iostream>

/*=============================================================================
  IMAGE(int N, MPI_Comm comm, root=0) - constructor which uses a
  duplicate communicator

  int N - the number of pixels in the image
  MPI_Comm comm - communicator which will be duplicated inside this image.
  int root - the root rank of the image. 
  ============================================================================*/
IMAGE::IMAGE(int N, MPI_Comm comm, int root){
  MPI_Comm dupComm;
  duplicateCommunicator=true;
  MPI_Comm_dup(comm,&dupComm);
  initialize(N,dupComm,root);
}


/*=============================================================================
  IMAGE(MPI_Comm comm, int N, root=0) - constructor which uses the
  communicator, not its duplicate.

  MPI_Comm comm - the MPI communicator to form the image across. The
  image will use this communicator, not a duplicate of it.
  int N - the number of pixels in the image.
  int root - the root node of the image. 
  ============================================================================*/
IMAGE::IMAGE(MPI_Comm comm, int N, int root){
  duplicateCommunicator=false;
  initialize(N,comm,root);
}


/*=============================================================================
  ~IMAGE() - destructor
  ============================================================================*/
IMAGE::~IMAGE(){
  if(duplicateCommunicator)
    MPI_Comm_free(&comm);
}


/*=============================================================================
  void getSize(int &N) - get the total size of the image
  ============================================================================*/
void IMAGE::getSize(int &N) const{
  N=IMAGE::N;
}


/*=============================================================================
  void setValue(int index, double v) - set the value of a pixel
  identified by its local index.
  ============================================================================*/
void IMAGE::setValue(int index, double v){
  data[index]=v;
}


/*=============================================================================
  double getValue(int index, double v) - get the value of a pixel
  identified by its local index.
  ============================================================================*/
double IMAGE::getValue(int index) const{
  return data[index];
}


/*=============================================================================
  int nIndex() - return the number of pixels stored in this rank
  ============================================================================*/
int IMAGE::nIndex() const{
  return nIndex(rank);
}


/*=============================================================================
  int indexToPixel(int index) - maps the local index for this rank to
  pixel (which is global index)
  ============================================================================*/
int IMAGE::indexToPixel(int index) const{
  return indexToPixel(rank,index);
}


/*=============================================================================
  IMAGE &operator=(IMAGE &img) - assignment operator

  This will copy the entire data area of each rank to the data area of
  this rank. The *this and img must be at least congruent
  (MPI_CONGRUENT or MPI_IDENT) or the function will exit with a
  failure. While this communicators of the images must be at least
  congruent the images need not be of the same size. The *this image
  is resized to match img. The communicator of *this image is not
  changed.
  ============================================================================*/
IMAGE &IMAGE::operator=(IMAGE &img){
  int result;
  MPI_Comm_compare(comm,img.comm,&result);
  if(result!=MPI_IDENT&&result!=MPI_CONGRUENT){
    std::cout << "IMAGE::operator=: error: the two IMAGES' communicators "
	      << "are neither identical nor congruent" << std::endl;
    exit(1);
  }

  N=img.N;
  time=img.time;
  pos=img.pos;
  roll=img.roll;
  pitch=img.pitch;
  yaw=img.yaw;
  data=img.data;
    
  return *this;
}


/*=============================================================================
  IMAGE &operator*=(double f) - multiply an image by a constant

  Multiply only pixels assigned to the local rank. 
  ============================================================================*/
IMAGE &IMAGE::operator*=(double f){
  int n=nIndex();
  for(int i=0;i<n;i++)
    data[i]*=f;
  
  return *this;
}


/*=============================================================================
  void setTime(aTime t) - set the time of the image
  ============================================================================*/
void IMAGE::setTime(aTime t){
  time=t;
}


/*=============================================================================
  aTime getTime() - get the time of the image
  ============================================================================*/
aTime IMAGE::getTime() const{
  return time;
}


/*=============================================================================
  void setPos(aVec p) - set the position of the image
  ============================================================================*/
void IMAGE::setPos(aVec p){
  pos=p;
}


/*=============================================================================
  aVec getPos() - get the position of the image
  ============================================================================*/
aVec IMAGE::getPos() const{
  return pos;
}


/*=============================================================================
  void setDir(double roll, double pitch, double yaw) - set the look
  direction angles for the image.
  ============================================================================*/
void IMAGE::setDir(double r, double p, double y){
  roll=r;
  pitch=p;
  yaw=y;
}


/*=============================================================================
  void getDir(double roll, double pitch, double yaw) - get the look
  direction angle for the image
  ============================================================================*/
void IMAGE::getDir(double &r, double &p, double &y) const{
  r=roll;
  p=pitch;
  y=yaw;
}


/*=============================================================================
  int write(FILE *fp) - write the image to a open file pointer

  The image is gathered at rank==root and rank==root then writes to
  the file. rank==root will broadcast the number of bytes written to
  all ranks so all ranks return the number of bytes written.

  The argument *fp is only used on rank root. 
  ============================================================================*/
int IMAGE::write(FILE *fp) const{
  int nn=0;
  
  if(rank==root)
    nn+=fwrite(&N,sizeof(int),1,fp);
  MPI_Bcast(&nn,1,MPI_INT,root,comm);
  
  nn+=writeData(fp);

  return nn;
}


/*=============================================================================
  int read(FILE *fp) - read the image from a open file pointer. 

  The image is read by rank==root. The image meta data are then
  broadcast to all ranks followed by the data scattered to all ranks'
  data. The number of bytes read by rank==root is broadcast to all
  ranks such that all ranks return the number of bytes read.
  ============================================================================*/
int IMAGE::read(FILE *fp){
  int nn=0;
  
  if(rank==root)
    nn+=fread(&N,sizeof(int),1,fp);
  MPI_Bcast(&N,1,MPI_INT,root,comm);
  MPI_Bcast(&nn,1,MPI_INT,root,comm);

  nn+=readData(fp);
  
  return nn;
}


/*=============================================================================
  int zwrite(gzFile fp) - write the image to a open gzip file pointer

  rank=0 writes the image stored at rank=0 to a file.
  rank=0 returns number of bytes written, other ranks return 0.
  ============================================================================*/
int IMAGE::zwrite(gzFile fp) const{
  std::vector<double> xdata;
  int nn=0;
  
  if(rank==root)
    nn+=gzwrite(fp,&N,sizeof(int));
  MPI_Bcast(&nn,1,MPI_INT,root,comm);

  nn+=zwriteData(fp);

  return nn;
}


/*=============================================================================
  int zread(gzFile fp) - read the image from a open gzip file pointer

  The image is read by rank==root. The image meta data are then
  broadcast to all ranks followed by the data scattered to all ranks'
  data. The number of bytes read by rank==root is broadcast to all
  ranks such that all ranks return the number of bytes read.
  ============================================================================*/
int IMAGE::zread(gzFile fp){
  int nn=0;

  if(rank==root)
    nn+=gzread(fp,&N,sizeof(int));
  MPI_Bcast(&N,1,MPI_INT,root,comm);
  MPI_Bcast(&nn,1,MPI_INT,root,comm);

  nn+=zreadData(fp);

  return nn;
}


/******************************************************************************
 ****************************** protected *************************************
 ******************************************************************************/


/*=============================================================================
  int indexToPixel(rank, int index) - maps the process rank and
  local index to pixel number (which is global index)
  ============================================================================*/
int IMAGE::indexToPixel(int rank, int index) const{
  return rank+index*comm_size;  
}


/*=============================================================================
  int writeData(FILE *fp) - write the data (all but N) to file pointer
  fp. Will write in rank==root and return number of bytes written in
  all ranks.
  ============================================================================*/
int IMAGE::writeData(FILE *fp) const{
  std::vector<double> xdata;
  int nn=0;

  gather(xdata);

  if(rank==root){
    nn+=fwrite(&roll,sizeof(double),1,fp);
    nn+=fwrite(&pitch,sizeof(double),1,fp);
    nn+=fwrite(&yaw,sizeof(double),1,fp);
    double x=pos.X(),y=pos.Y(),z=pos.Z();
    nn+=fwrite(&x,sizeof(double),1,fp);
    nn+=fwrite(&y,sizeof(double),1,fp);
    nn+=fwrite(&z,sizeof(double),1,fp);
    int yr,mo,dy,hr,mn,se;
    long ns;
    time.get(yr,mo,dy,hr,mn,se,ns);
    nn+=fwrite(&yr,sizeof(int),1,fp);
    nn+=fwrite(&mo,sizeof(int),1,fp);
    nn+=fwrite(&dy,sizeof(int),1,fp);
    nn+=fwrite(&hr,sizeof(int),1,fp);
    nn+=fwrite(&mn,sizeof(int),1,fp);
    nn+=fwrite(&se,sizeof(int),1,fp);
    nn+=fwrite(&ns,sizeof(long),1,fp);
    nn+=fwrite(&xdata[0],sizeof(double),N,fp);    
  }
  
  MPI_Bcast(&nn,1,MPI_INT,root,comm);

  return nn;
}


/*=============================================================================
  int readData(FILE *fp) - read the data (all but N) from file pointer
  fp. This will read in rank==root and return number of bytes written
  in all ranks.

  Important: before calling this function N must be read and broadcast
  to all ranks.
  ============================================================================*/
int IMAGE::readData(FILE *fp){
  std::vector<double> xdata;
  int nn=0;

  if(rank==root){
    nn+=fread(&roll,sizeof(double),1,fp);
    nn+=fread(&pitch,sizeof(double),1,fp);
    nn+=fread(&yaw,sizeof(double),1,fp);
    double x,y,z;
    nn+=fread(&x,sizeof(double),1,fp);
    nn+=fread(&y,sizeof(double),1,fp);
    nn+=fread(&z,sizeof(double),1,fp);
    pos=aVec(x,y,z);
    int yr,mo,dy,hr,mn,se;
    long ns;
    nn+=fread(&yr,sizeof(int),1,fp);
    nn+=fread(&mo,sizeof(int),1,fp);
    nn+=fread(&dy,sizeof(int),1,fp);
    nn+=fread(&hr,sizeof(int),1,fp);
    nn+=fread(&mn,sizeof(int),1,fp);
    nn+=fread(&se,sizeof(int),1,fp);
    nn+=fread(&ns,sizeof(long),1,fp);
    time.set(yr,mo,dy,hr,mn,se,ns);
    xdata.resize(maxNIndex());
    nn+=fread(&xdata[0],sizeof(double),N,fp);
  }

  bcastMetadata();
  scatter(xdata);

  MPI_Bcast(&nn,1,MPI_INT,root,comm);

  return nn;
}


/*=============================================================================
  int zwriteData(gzFile fp) - write the data (all but N) to file
  pointer fp. This will write in any rank where it is called, thus it
  should only be called by rank==root.

  returns number of bytes written.
  ============================================================================*/
int IMAGE::zwriteData(gzFile fp) const{
  std::vector<double> xdata;
  int nn=0;

  gather(xdata);

  if(rank==root){
    nn+=gzwrite(fp,&roll,sizeof(double));
    nn+=gzwrite(fp,&pitch,sizeof(double));
    nn+=gzwrite(fp,&yaw,sizeof(double));
    double x=pos.X(),y=pos.Y(),z=pos.Z();
    nn+=gzwrite(fp,&x,sizeof(double));
    nn+=gzwrite(fp,&y,sizeof(double));
    nn+=gzwrite(fp,&z,sizeof(double));
    int yr,mo,dy,hr,mn,se;
    long ns;
    time.get(yr,mo,dy,hr,mn,se,ns);
    nn+=gzwrite(fp,&yr,sizeof(int));
    nn+=gzwrite(fp,&mo,sizeof(int));
    nn+=gzwrite(fp,&dy,sizeof(int));
    nn+=gzwrite(fp,&hr,sizeof(int));
    nn+=gzwrite(fp,&mn,sizeof(int));
    nn+=gzwrite(fp,&se,sizeof(int));
    nn+=gzwrite(fp,&ns,sizeof(long));
    nn+=gzwrite(fp,&data[0],sizeof(double)*N);
  }    
  
  MPI_Bcast(&nn,1,MPI_INT,root,comm);
  
  return nn;
}


/*=============================================================================
  int zreadData(gzFile fp) - read the data (all but N) from file
  pointer fp. This will read in any rank where it is called, thus it
  should only be called by rank==root.

  returns number of bytes written.
  ============================================================================*/
int IMAGE::zreadData(gzFile fp){
  std::vector<double> xdata;
  int nn=0;

  if(rank==root){
    nn+=gzread(fp,&roll,sizeof(double));
    nn+=gzread(fp,&pitch,sizeof(double));
    nn+=gzread(fp,&yaw,sizeof(double));
    double x,y,z;
    nn+=gzread(fp,&x,sizeof(double));
    nn+=gzread(fp,&y,sizeof(double));
    nn+=gzread(fp,&z,sizeof(double));
    pos=aVec(x,y,z);
    int yr,mo,dy,hr,mn,se;
    long ns;
    nn+=gzread(fp,&yr,sizeof(int));
    nn+=gzread(fp,&mo,sizeof(int));
    nn+=gzread(fp,&dy,sizeof(int));
    nn+=gzread(fp,&hr,sizeof(int));
    nn+=gzread(fp,&mn,sizeof(int));
    nn+=gzread(fp,&se,sizeof(int));
    nn+=gzread(fp,&ns,sizeof(long));
    time.set(yr,mo,dy,hr,mn,se,ns);
    xdata.resize(maxNIndex());
    nn+=gzread(fp,&data[0],sizeof(double)*N);
  }

  bcastMetadata();
  scatter(xdata);

  return nn;
}


/******************************************************************************
 ***************************** Private functions ******************************
 ******************************************************************************/


/*=============================================================================
  void initialize(int N, MPI_Comm comm, int root) - initialization
  ============================================================================*/
void IMAGE::initialize(int N, MPI_Comm comm, int root){
  IMAGE::N=N;
  IMAGE::comm=comm;
  IMAGE::root=root;
  MPI_Comm_size(comm,&comm_size);
  MPI_Comm_rank(comm,&rank);
  
  data.resize(maxNIndex());
}


/*=============================================================================
  int nIndex(int rank) - return the number of pixels stored in a
  specified rank.
  ============================================================================*/
int IMAGE::nIndex(int rank) const{
  return (N+comm_size-1-rank)/comm_size;
}


/*=============================================================================
  int maxNIndex() - return the largest number of pixels stored in
  any rank. This value defines the size of internal communications.
  ============================================================================*/
int IMAGE::maxNIndex() const{
  if(N%comm_size==0)
    return N/comm_size;
  return N/comm_size+1;
}


/*=============================================================================
  void gather(std::vector<double> &xdata) - gathers the pixel data from
  all ranks and correctly organizes it in xdata on rank==root (xdata
  resized as necessary). At other ranks xdata is not modified.
  ============================================================================*/
void IMAGE::gather(std::vector<double> &xdata) const{
  int maxIndex=maxNIndex();
  std::vector<double> buffer;
  if(rank==0)
    buffer.resize(comm_size*maxIndex);
  MPI_Gather((void*)&data[0],maxIndex,MPI_DOUBLE,&buffer[0],maxIndex,
	     MPI_DOUBLE,root,comm);
  if(rank==root){
    xdata.resize(N);
    for(int iRank=0;iRank<comm_size;iRank++){
      int nIdx=nIndex(iRank);
      for(int iIdx=0;iIdx<nIdx;iIdx++)
	xdata[indexToPixel(iRank,iIdx)]=buffer[iRank*maxIndex+iIdx];
    }
  }
}


/*=============================================================================
  void scatter(std::vector<double> &xdata) - scatters the pixel data in
  xdata on rank==root correctly to the other ranks' data. xdata on
  other ranks are ignored.
  ============================================================================*/
void IMAGE::scatter(const std::vector<double> &xdata){
  int maxIndex=maxNIndex();
  std::vector<double> buffer;
  if(rank==0)
    buffer.resize(comm_size*maxIndex);
  
  if(rank==root){
    for(int iRank=0;iRank<comm_size;iRank++){
      int nIdx=nIndex(iRank);
      for(int iIdx=0;iIdx<nIdx;iIdx++)
	buffer[iRank*maxIndex+iIdx]=xdata[indexToPixel(iRank,iIdx)];
    }
  }
  
  MPI_Scatter(&buffer[0],maxIndex,MPI_DOUBLE,&data[0],maxIndex,MPI_DOUBLE,
	      root,comm);
}


/*=============================================================================
  void bcastMetadata() - broadcast the metadata stored on rank root to
  all ranks, and also resizes the data area on each rank to match. 

  This is implemented as a large number of broadcasts of basic data
  types. It could probably be implemented more efficiently as a single
  broadcast of a compount data type. However this is a rare call so
  optimizing it is not a high priority yet.
  ============================================================================*/
void IMAGE::bcastMetadata(){
  MPI_Bcast(&N,1,MPI_INT,root,comm);
  MPI_Bcast(&roll,1,MPI_DOUBLE,root,comm);
  MPI_Bcast(&pitch,1,MPI_DOUBLE,root,comm);
  MPI_Bcast(&yaw,1,MPI_DOUBLE,root,comm);
  double x=pos.X(),y=pos.Y(),z=pos.Z();
  MPI_Bcast(&x,1,MPI_DOUBLE,root,comm);
  MPI_Bcast(&y,1,MPI_DOUBLE,root,comm);
  MPI_Bcast(&z,1,MPI_DOUBLE,root,comm);
  pos=aVec(x,y,z);
  int yr,mo,dy,hr,mn,se;
  long ns;
  time.get(yr,mo,dy,hr,mn,se,ns);
  MPI_Bcast(&yr,1,MPI_INT,root,comm);
  MPI_Bcast(&mo,1,MPI_INT,root,comm);
  MPI_Bcast(&dy,1,MPI_INT,root,comm);
  MPI_Bcast(&hr,1,MPI_INT,root,comm);
  MPI_Bcast(&mn,1,MPI_INT,root,comm);
  MPI_Bcast(&se,1,MPI_INT,root,comm);
  MPI_Bcast(&ns,1,MPI_LONG,root,comm);
  time.set(yr,mo,dy,hr,mn,se,ns);

  data.resize(maxNIndex());
}


/******************************************************************************
 ***************************** friend functions *******************************
 ******************************************************************************/


/*=============================================================================
  double average(IMAGE &i) - compute the average value per pixel of an image

  The sum is computed at each rank, summed in a reduceall, and average
  computed and returned at all ranks.
  ============================================================================*/
double average(const IMAGE &image){
  double sum=0,allsum;
  int n=image.nIndex();
  for(int i=0;i<n;i++)
    sum+=image.data[i];
  
  MPI_Allreduce(&sum,&allsum,1,MPI_DOUBLE,MPI_SUM,image.comm);
  return allsum/image.N;
}


/*====================================================================
  double absdiff(IMAGE &i1, IMAGE &i2) - compute the average absolute
  difference between two images.

  The computation is distributed with each rank computing its part of
  the image and then combined. The same end result is returned by all
  ranks.
  ===================================================================*/
double absdiff(const IMAGE &i1, const IMAGE &i2){
  double sum=0,allsum;
  int n=i1.nIndex();
  for(int i=0;i<n;i++)
    sum+=fabs(i1.data[i]-i2.data[i]);
  MPI_Allreduce(&sum,&allsum,1,MPI_DOUBLE,MPI_SUM,i1.comm);
  allsum/=i1.N;
  return allsum;
}


/*====================================================================
  double rmsdiff(IMAGE &i1, IMAGE &i2) - compute the RMS difference
  between two images.

  The computation is distributed with each rank computing its part of
  the image and then combined. The same end result is returned by all
  ranks.
  ===================================================================*/
double rmsdiff(const IMAGE &i1, const IMAGE &i2){
  double diff,sum=0,allsum;
  int n=i1.nIndex();
  for(int i=0;i<n;i++){
    diff=i1.data[i]-i2.data[i];
    sum+=diff*diff;
  }
  
  MPI_Allreduce(&sum,&allsum,1,MPI_DOUBLE,MPI_SUM,i1.comm);
  allsum=sqrt(allsum/i1.N);
  return allsum;
}


/*====================================================================
  double poidiff(IMAGE &i1, IMAGE &i2) - compute the Poisson
  difference between two images. Use image 1 as a mean counts image
  and from it compute the probability of the other image. This
  function returns the negative of the log of the probability. This
  will be a positive and usully large number.

  The computation is distributed with each rank computing its part of
  the image and then combined. The same end result is returned by all
  ranks.
  ===================================================================*/
double poidiff(const IMAGE &i1, const IMAGE &i2){
  double sum=0,allsum;
  int n=i1.nIndex();
  for(int i=0;i<n;i++)
    sum+=log(gsl_ran_poisson_pdf((int)round(i2.data[i]),
				 i1.data[i]));
  MPI_Allreduce(&sum,&allsum,1,MPI_DOUBLE,MPI_SUM,i1.comm);
  return -allsum;
}
