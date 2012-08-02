#ifndef bmia_TensorColoring_h
#define bmia_TensorColoring_h

#include "ScalarMeasures.h"

namespace bmia {
namespace TensorColoring {

static const int TypeLUT = 0;			// Use Color lookup table
static const int TypeMEV = 1;			// RGB coloring of MEV, nog weighted by measure
static const int TypeWeightedMEV = 2;		// RGB coloring of MEV, weighted by measure
static const int num_coloring_types = 3;

} // namespace TensorColoring
} // namespace bmia
#endif // bmia_TensorColoring_h
