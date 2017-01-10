/******************************************************************************
 * This is class RIMAGE. It contains a rectangular image output by a the      *
 * rectangular camera RCAMERA.                                                *
 ******************************************************************************/

#include "../include/rimage.H"

/*=============================================================================
  RIMAGE(int nX, int nY, MPI_Comm comm, int root=0) - constructor which
  uses a duplicate of the provided communicator.
  
  int nX, nY - image size, nX is in the horizontal/azimuthal direction
  and nY is in the vertical/theta direction.
  MPI_Comm comm - the communicator to duplicate.
  int root - the root rank for writing and reading images and gather scatter 
  operations.
  ============================================================================*/
RIMAGE::RIMAGE(int nX, int nY, MPI_Comm comm, int root)
  :IMAGE(nX*nY,comm,root){
  initialize(nX,nY);
}


/*=============================================================================
  RIMAGE(MPI_Comm comm, int nX, int nY, int root=0) - constructor
  which uses the provided communicator

  MPI_Comm comm - the communicator to use.
  int nX, nY - image size, nX is in the horizontal/azimuthal direction
  and nY is in the vertical/theta direction.
  int root - the root rank for writing and reading images and gather scatter 
  operations.
  ============================================================================*/
RIMAGE::RIMAGE(MPI_Comm comm, int nX, int nY, int root)
  :IMAGE(comm,nX*nY,root){
  initialize(nX,nY);
}


/*=============================================================================
  RIMAGE::~RIMAGE() - destructor
  ============================================================================*/
RIMAGE::~RIMAGE(){

}


/*=============================================================================
  void getSize(int &nX, int &nY) - get the total size of the image
  ============================================================================*/
void RIMAGE::getSize(int &nX, int &nY) const{
  nX=RIMAGE::nX;
  nY=RIMAGE::nY;
}


/*=============================================================================
  void indexToPixel(int index, int &iX, int &iY) - returns the pixel
  image coordinate for the index pixel stored in this rank.
  ============================================================================*/
void RIMAGE::indexToPixel(int index, int &iX, int &iY) const{
  return indexToPixel(rank,index,iX,iY);
}


/*=============================================================================
  RIMAGE &RIMAGE::operator=(RIMAGE &img) - assignment operator

  Calls IMAGE::operator= and then copies the data specific to RIMAGE.
  ============================================================================*/
RIMAGE &RIMAGE::operator=(RIMAGE &img){
  IMAGE::operator=(img);
  nX=img.nX;
  nY=img.nY;

  return *this;
}
  

/*=============================================================================
  int write(FILE *fp) - writes an image to a file pointer

  Only rank==root writes. This call is implemented by rank==root
  writing nX and nY to file, then broadcasting the number of bytes
  written. Then calling writeData for the rest of the data. 

  All ranks return the total number of bytes written.
  ============================================================================*/
int RIMAGE::write(FILE *fp) const{
  int nn=0;

  if(rank==root){
    nn+=fwrite(&nX,sizeof(int),1,fp);
    nn+=fwrite(&nY,sizeof(int),1,fp);
  }
  MPI_Bcast(&nn,1,MPI_INT,root,comm);

  nn+=writeData(fp);
  
  return nn;
}


/*=============================================================================
  int read(FILE *fp) - read the image from a open file pointer

  This is implemented by reading nX and nY at rank==root, then
  broadcasting those from rank==root to all ranks, then broadcasting
  the number of bytes read. Then calling readData to read the rest of
  the data. All ranks return the total number of bytes read.
  ============================================================================*/
int RIMAGE::read(FILE *fp){
  int nn=0;
  
  if(rank==root){
    nn+=fread(&nX,sizeof(int),1,fp);
    nn+=fread(&nY,sizeof(int),1,fp);
  }
  MPI_Bcast(&nX,1,MPI_INT,root,comm);
  MPI_Bcast(&nY,1,MPI_INT,root,comm);
  MPI_Bcast(&nn,1,MPI_INT,root,comm);

  nn+=readData(fp);

  return nn;
}


/*=============================================================================
  int zwrite(gzFile fp) - writes an image to a file pointer

  Only rank==root writes. This call is implemented by rank==root
  writing nX and nY to file, then broadcasting the number of bytes
  written. Then calling zwriteData for the rest of the data. 

  All ranks return the total number of bytes written.
  ============================================================================*/
int RIMAGE::zwrite(gzFile fp) const{
  int nn=0;

  if(rank==root){
    nn+=gzwrite(fp,&nX,sizeof(int));
    nn+=gzwrite(fp,&nY,sizeof(int));
  }
  MPI_Bcast(&nn,1,MPI_INT,root,comm);

  nn+=zwriteData(fp);
  
  return nn;
}


/*=============================================================================
  int zread(gzFile fp) - read the image from a open file pointer

  This is implemented by reading nX and nY at rank==root, then
  broadcasting those from rank==root to all ranks, then broadcasting
  the number of bytes read. Then calling zreadData to read the rest of
  the data. All ranks return the total number of bytes read.
  ============================================================================*/
int RIMAGE::zread(gzFile fp){
  int nn=0;
  
  if(rank==root){
    nn+=gzread(fp,&nX,sizeof(int));
    nn+=gzread(fp,&nY,sizeof(int));
  }
  MPI_Bcast(&nX,1,MPI_INT,root,comm);
  MPI_Bcast(&nY,1,MPI_INT,root,comm);
  MPI_Bcast(&nn,1,MPI_INT,root,comm);

  nn+=zreadData(fp);

  return nn;
}


/******************************************************************************
 ********************** protected functions ***********************************
 ******************************************************************************/



/******************************************************************************
 ********************** private functions *************************************
 ******************************************************************************/


/*=============================================================================
  void initialize(int nX, int nY) - initialize
  ============================================================================*/
void RIMAGE::initialize(int nX, int nY){
  RIMAGE::nX=nX;
  RIMAGE::nY=nY;
}


/*=============================================================================
  void indexToPixel(int rank, int index, int &iX, int &iY) - returns the
  pixel image coordinate for a index on a specified rank.
  ============================================================================*/
void RIMAGE::indexToPixel(int rank, int index, int &iX, int &iY) const{
  int k=IMAGE::indexToPixel(rank,index);
  iY=k/nX;
  iX=k%nX;
}


/*=============================================================================
  void bcastMetadata() - broadcasts metadata from rank==root to all ranks. 

  Only broadcasts data which are in RIMAGE but not in IMAGE.
  ============================================================================*/
void RIMAGE::bcastMetadata(){
  MPI_Bcast(&nX,1,MPI_INT,root,comm);
  MPI_Bcast(&nY,1,MPI_INT,root,comm);
}

