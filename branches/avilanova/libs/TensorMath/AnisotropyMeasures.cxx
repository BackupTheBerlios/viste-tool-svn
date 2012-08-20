/**
 * AnisotropyMeasures.cxx
 * by Tim Peeters
 *
 * 2006-12-26	Tim Peeters
 * - First version
 *
 * 2008-02-03	Jasper Levink
 * - Added single eigenvalues
 *
 */

#include "AnisotropyMeasures.h"
#include <assert.h>
#include <math.h>

namespace bmia {
namespace AnisotropyMeasures {

double AnisotropyMeasure(int measure, double evals[3])
{
  assert( 0 <= measure );
  assert( measure < AnisotropyMeasures::numberOfMeasures );

  double result;
  switch (measure)
    {
    case FA:
      result = AnisotropyMeasures::FractionalAnisotropy(evals);
      break;
    case RA:
      result = AnisotropyMeasures::RelativeAnisotropy(evals);
      break;
    case Cl:
      result = AnisotropyMeasures::LinearAnisotropy(evals);
      break;
    case Cp:
      result = AnisotropyMeasures::PlanarAnisotropy(evals);
      break;
    case Cs:
      result = AnisotropyMeasures::Isotropy(evals);
      break;
    case Ca:
      result = AnisotropyMeasures::Anisotropy(evals);
      break;
    case MD:
      result = AnisotropyMeasures::MeanDiffusivity(evals);
      break;
    case L1:
      result = AnisotropyMeasures::EigenValue(1,evals);
      break;
    case L2:
      result = AnisotropyMeasures::EigenValue(2,evals);
      break;
    case L3:
      result = AnisotropyMeasures::EigenValue(3,evals);
      break;
    default:
      // should never happen
      result = 0.0;
      assert( false );
    } // switch
  return result;
}

const char* GetLongName(int measure)
{
  assert( 0 <= measure );
  assert( measure <= AnisotropyMeasures::numberOfMeasures);

  return longNames[measure];
}

const char* GetShortName(int measure)
{
  assert( 0 <= measure );
  assert( measure <= AnisotropyMeasures::numberOfMeasures);

  return shortNames[measure];
}

double FractionalAnisotropy(double evals[3])
{
  double d = evals[0]*evals[0] + evals[1]*evals[1] +
            evals[2]*evals[2];
  if (d == 0.0)
    {
    return 0.0;
    }

  double r; // root of the sum of the squared differences.
  r = rsds(evals[0], evals[1], evals[2]);

  return r / sqrt(2.0*d);
}

double RelativeAnisotropy(double evals[3])
{
  double sum = evals[0] + evals[1] + evals[2];
  if (sum == 0.0)
    {
    return 0.0;
    }

  double r = rsds(evals[0], evals[1], evals[2]);

  return r/(sqrt(2.0)*sum);
}

// root of the sum of the squared differences.
double rsds(double u, double v, double w)
{
  double d12 = v-u;
  double d23 = w-v;
  double d13 = w-u;
  double sumsq = d12*d12 + d23*d23 + d13*d13;
  // sumsq >= 0.
  return sqrt(sumsq);
}

double LinearAnisotropy(double evals[3])
{
  double sum = evals[0] + evals[1] + evals[2];
  if (sum == 0.0)
    {
    return 0.0;
    }
  return (evals[0]-evals[1])/sum;
}

double PlanarAnisotropy(double evals[3])
{
  double sum = evals[0] + evals[1] + evals[2];
  if (sum == 0.0)
    {
    return 0.0;
    }
  return 2*(evals[1]-evals[2])/sum;
}

double Isotropy(double evals[3])
{
  double sum = evals[0] + evals[1] + evals[2];
  if (sum == 0.0)
    {
    return 0.0;
    }
  return 3*evals[2]/sum;
}

double Anisotropy(double evals[3])
{
  return 1.0 - Isotropy(evals);
}

double MeanDiffusivity(double evals[3])
{
  return (evals[0]+evals[1]+evals[2])/3.0;
}

double EigenValue(int eigenValue, double evals[3])
{
  if(eigenValue == 1 || eigenValue == 2 || eigenValue == 3)
	return evals[eigenValue-1];
  else // Should never happen
	return 0.0;
}

} // namespace AnisotropyMeasures
} // namespace bmia
