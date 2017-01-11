/******************************************************************************
 * This is class LOSGSLQAG. It implements line-of-sight integration using the *
 * GNU Scientific Library QAG integration function.                           *
 ******************************************************************************/

#ifndef _LOSGSLQAG_H_
#define _LOSGSLQAG_H_

#include "los.H"

#include <gsl/gsl_integration.h>

class LOSGSLQAG: public LOS{
public:
  LOSGSLQAG();
  ~LOSGSLQAG();
  virtual double val(const aVec &p) const =0;
  virtual double los(const aVec &p, const aVec &d) const;
private:
  size_t nWorkspace;
  gsl_integration_workspace *workspace;
  double epsabs;
  double epsrel;
  virtual double losStart(const aVec &s, const aVec &d) const =0;
  virtual double losStop(const aVec &s, const aVec &d) const =0;
  virtual double losStep(const aVec &s, const aVec &d) const =0;
};

struct LOSGSLQAG_params{
  aVec start;
  aVec dir;
  const LOSGSLQAG *integrand;
};

double LOSGSLQAG_model(double x, void *pp);

#endif



