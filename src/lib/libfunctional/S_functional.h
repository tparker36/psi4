#ifndef S_functional_h
#define S_functional_h
/**********************************************************
* S_functional.h: declarations for S_functional for KS-DFT
* Robert Parrish, robparrish@gmail.com
* 09/01/2010
*
***********************************************************/
#include <libmints/properties.h>
#include <libciomr/libciomr.h>
#include "functional.h"
#include <stdlib.h>
#include <string>
#include <vector>

using namespace psi;
using namespace boost;
using namespace std;

namespace psi { namespace functional {

class S_Functional : public Functional {
public:
    S_Functional(int npoints, int deriv);
    ~S_Functional();
    void init();
    void computeRKSFunctional(shared_ptr<Properties> prop);
    void computeUKSFunctional(shared_ptr<Properties> prop);
};
}}
#endif

