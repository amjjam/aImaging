/******************************************************************************
 * This is class IMAGE. It contains an image output by a camera together with *
 * associated metadata.                                                       *
 ******************************************************************************/

#ifndef _IMAGE_H_
#define _IMAGE_H_

#include <aTime.H>
#include <aVec.H>
#include <vector>

#include <stdio.h>
#include <zlib.h>

#include <mpi.h>

#include <gsl/gsl_randist.h>

class IMAGE{
public:
  IMAGE(int N, MPI_Comm comm, int root=0);
  IMAGE(MPI_Comm comm, int N, int root=0);
  ~IMAGE();
  void getSize(int &N) const;
  void setValue(int index, double v);
  double getValue(int index) const;
  int nIndex() const;
  int indexToPixel(int index) const;
  IMAGE &operator=(IMAGE &img);
  IMAGE &operator*=(double f);

  void setTime(aTime t);
  aTime getTime() const;
  void setPos(aVec p);
  aVec getPos() const;
  void setDir(double roll, double pitch, double yaw);
  void getDir(double &roll, double &pitch, double &yaw) const;
  
  virtual int write(FILE *fp) const;
  virtual int read(FILE *fp);
  virtual int zwrite(gzFile fp) const;
  virtual int zread(gzFile fp);

  friend double average(const IMAGE &i);
  friend double absdiff(const IMAGE &i1, const IMAGE &i2);
  friend double rmsdiff(const IMAGE &i1, const IMAGE &i2);
  friend double poidiff(const IMAGE &i1, const IMAGE &i2);
protected:
  int N;
  std::vector<double> data;
  MPI_Comm comm;
  int rank;  
  int root;
  int indexToPixel(int rank, int index) const;
  int writeData(FILE *fp) const;
  int readData(FILE *fp);
  int zwriteData(gzFile fp) const;
  int zreadData(gzFile fp);
private:
  void initialize(int N, MPI_Comm comm, int root);
  int nIndex(int rank) const;
  int maxNIndex() const;
  void gather(std::vector<double> &data) const;
  void scatter(const std::vector<double> &data);
  void bcastMetadata();
  aTime time;
  aVec pos;
  double roll,pitch,yaw;
  bool duplicateCommunicator;
  int comm_size;
};

double absdiff(const IMAGE &i1, const IMAGE &i2);
double rmsdiff(const IMAGE &i1, const IMAGE &i2);
double poidiff(const IMAGE &i1, const IMAGE &i2);
double average(const IMAGE &i);

#endif



