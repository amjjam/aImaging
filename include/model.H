/******************************************************************************
 * This is class MODEL. It is a base class for a model.                       *
 ******************************************************************************/

#ifndef _MODEL_H_
#define _MODEL_H_

class MODEL{
public:
  MODEL(){};
  ~MODEL(){};
  virtual double value(const aVec &p) const =0;
private:
  
};

#endif



