/**
 * AnisotropyMeasures.h
 * by Tim Peeters
 *
 * 2006-12-26	Tim Peeters
 * - First version
 * 
 * 2008-02-03	Jasper Levink
 * - Added single eigenvalues
 *
 */

#ifndef bmia_AnisotropyMeasures_h
#define bmia_AnisotropyMeasures_h

namespace bmia {

/**
 * To add a measure, edit the consts below, modify the implementation
 * of AnisotropyMeasure(int measure, double evals[3]), and implement
 * the computation of the measure itself. Also update numberOfMeasures.
 */
namespace AnisotropyMeasures {

//public:
  //static const int first_index = 300;
  static const int FA = 0;
  static const int RA = 1;
  static const int Cl = 2;
  static const int Cp = 3;
  static const int Cs = 4;
  static const int Ca = 5;
  static const int MD = 6;
  static const int L1 = 7;
  static const int L2 = 8;
  static const int L3 = 9;
  static const int numberOfMeasures = 10;
  //static const int stop_index = first_index + numberOfMeasures;
  //static const int last_index = stop_index - 1;


  static const char * longNames[] = {
	"Fractional Anisotropy",
	"Relative Anisotropy",
	"Linear Anisotropy",
	"Planar Anisotropy",
	"Isotropy",
	"Anisotropy",
	"Mean Diffusivity",
	"First Eigenvalue",
	"Second Eigenvalue",
	"Third Eigenvalue"
  }; // longNames

  static const char* shortNames[] = {
	  "FA",
	  "RA",
	  "Cl",
	  "Cp",
	  "Cs",
	  "Ca",
	  "MD",
	  "L1",
	  "L2",
	  "L3"	
  }; // shortNames

  /**
   * Return the value of the specified measure with the given
   * eigenvalues. 0 <= measure < numberOfMeasures.
   */
  double AnisotropyMeasure(int measure, double evals[3]);

  /**
   * Return the long/short name of the specified measure.
   */
  const char* GetLongName(int measure);
  const char* GetShortName(int measure);

  /**
   * Computes and returns the Fractional Anisotopy from the specified eigenvalues.
   * If the sum of the squares of the three parameters is 0, then 0 is returned.
   */
  double FractionalAnisotropy(double evals[3]);

  /**
   * Computes and returns the Relative Anisotropy from the three specified eigenvalues.
   * If the sum of the eigenvalues is 0, then 0 is returned.
   */
  double RelativeAnisotropy(double evals[3]);

  /**
   * Returns the root of the sum of the square differences between the
   * pairs of the three values given. Used by FractionalAnisotropy() and
   * RelativeAnisotropy().
   */
  double rsds(double u, double v, double w);

  /**
   * Computes linear, planar, or spherical anisotropy indices from the given
   * triplet of eigenvalues.
   * If the sum of the eigenvalues is 0, then 0 is returned.
   */
  double LinearAnisotropy(double evals[3]);
  double PlanarAnisotropy(double evals[3]);
  double Isotropy(double evals[3]);
  double Anisotropy(double evals[3]);

  /**
   * Return mean diffusivity
   */
  double MeanDiffusivity(double evals[3]);

  /**
   * Return single eigenvalue (int eigenValue = 1|2|3)
   */
  double EigenValue(int eigenValue,double evals[3]);

}; // namespace AnisotropyMeasures

} // namespace bmia

#endif // bmia_AnisotropyMeasures_h
