/******************************************************************************
 * This is base class LOS. It defines line-of-sight integration.              *
 ******************************************************************************/

#ifndef _LOS_H_
#define _LOS_H_

#include <aVec.H>

class LOS{
public:
  LOS(){};
  ~LOS(){};
  virtual double los(const aVec &p, const aVec &d) const =0;
private:
  
};

#endif
