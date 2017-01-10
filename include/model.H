/******************************************************************************
 * This is the base class of the model. It can return brightness at a point   *
 * and the integrated brightness along a ray path.                            *
 ******************************************************************************/

#ifndef _MODEL_H_
#define _MODEL_H_

#include <aVec.H>

#include <gsl/gsl_integration.h>

class MODEL{
public:
  MODEL();
  ~MODEL();
  virtual double brightness(const aVec &p) const =0;
  virtual double losStart(const aVec &s, const aVec &d) const =0;
  virtual double losStop(const aVec &s, const aVec &d) const =0;
  virtual double losStep(const aVec &s, const aVec &d) const =0;
  virtual double los(const aVec &s, const aVec &d) const;
  //  virtual double brightness(double x) const =0;
  virtual size_t getLosNWorkspace() const;
  virtual void getLosEps(double &epsabs, double &epsrel) const;
  virtual void setLosNWorkspace(size_t nWorkspace);
  virtual void setLosEps(double epsabs, double epsrel);
protected:
  size_t nWorkspace;
  gsl_integration_workspace *workspace;
  double epsabs;
  double epsrel;
};

struct MODEL_gsl_params{
  aVec start;
  aVec dir;
  const MODEL *pt_MODEL;
};

double MODEL_gsl_brightness(double x, void *pp);

#endif
