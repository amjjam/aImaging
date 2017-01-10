/******************************************************************************
 * This is class LOSGSLQAG. It implements line-of-sight integration using the *
 * GNU Scientific Library QAG integration function.                           *
 ******************************************************************************/

#include "../include/losGslQag.H"


/*=============================================================================
  LOSGSLQAG() - constructor
  ============================================================================*/
LOSGSLQAG::LOSGSLQAG(){
  nWorkspace=1000;
  workspace=gsl_integration_workspace_alloc(nWorkspace);
  epsabs=0.1;
  epsrel=0.1;
}


/*=============================================================================
  ~LOSGSLQAG() - destructor
  ============================================================================*/
LOSGSLQAG::~LOSGSLQAG(){
  gsl_integration_workspace_free(workspace);
}


/*=============================================================================
  double los(const aVec &s, const aVec &d) - do a line of sight integral of
  brightness starting at position s in direction d.
  
  aVec &s - starting position
  aVec &d - direction vector
  ============================================================================*/
double LOSGSLQAG::los(const aVec &s, const aVec &dd) const{
  gsl_function F;
  struct LOSGSLQAG_params params;
  params.start=s;
  params.dir=unit(dd);
  params.integrand=this;
  F.function=&LOSGSLQAG_model;
  F.params=&params;
  double start=losStart(params.start,params.dir);
  double stop=losStop(params.start,params.dir);
  double sum,abserr;
  int error=0;
  if((error=gsl_integration_qag(&F,start,stop,epsabs,epsrel,nWorkspace,
				5,workspace,&sum,&abserr))>0)
    std::cout << "gsl_integration returned " << error << std::cout;
  
  return sum;
}


/*============================================================================
  double LOGGSLQAG_integrand(double x, void *pp) - interface function
  for GSL to the integrand.
  
  double x - position to evaluate the function at
  void *pp - pointer to parameters
  ===========================================================================*/
double LOSGSLQAG_integrand(double x, void *pp){
  LOSGSLQAG_params *p=(LOSGSLQAG_params *) pp;
  return p->integrand->val(p->start+p->dir*x);
}
