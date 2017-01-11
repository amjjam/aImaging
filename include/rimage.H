/******************************************************************************
 * This is class RIMAGE. It contains a rectangular image output by a the      *
 * rectangular camera RCAMERA.                                                *
 ******************************************************************************/

#ifndef _RIMAGE_H_
#define _RIMAGE_H_

#include "image.H"

class RIMAGE: public IMAGE{
public:
  RIMAGE(int nX, int nY, MPI_Comm comm, int root=0);
  RIMAGE(MPI_Comm comm, int nX, int nY, int root=0);
  ~RIMAGE();
  void getSize(int &nX, int &nY) const;
  void indexToPixel(int index, int &iX, int &iY) const;
  RIMAGE &operator=(RIMAGE &img);

  int write(FILE *fp) const;
  int read(FILE *fp);
  int zwrite(gzFile fp) const;
  int zread(gzFile fp);
protected:
  int nX,nY;
private:
  void initialize(int nX, int nY);
  void indexToPixel(int rank, int index, int &iX, int &iY) const;
  void bcastMetadata();
};

#endif



