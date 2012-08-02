/**
 * vtkDTIPlanesActor.h
 * by Tim Peeters
 *
 * 2005-11-14	Tim Peeters
 * - First version
 *
 * 2006-01-23	Tim Peeters
 * - Added SetColoring() function.
 *
 * 2006-01-24	Tim Peeters
 * - Added functions for setting interpolation.
 *
 * 2006-03-06	Tim Peeters
 * - Added Set{X,Y,Z}Visible functions.
 *
 * 2006-03-28	Tim Peeters
 * - Added vtkLookupTable.
 *
 * 2006-04-10	Tim Peeters
 * - Added SetLookupTable(vtkLookupTable* lut);
 * - Removed bmia::vtkTensorsToColors stuff.
 *
 * 2006-04-11	Tim Peeters
 * - Added SetDTILookupTables().
 *
 * 2006-04-12	Tim Peeters
 * - Remove filtering stuff. This is now in vtkDTIDataManager.
 * - Make input vtkDTIDataManager instead of vtkImageData.
 *
 * 2006-05-03	Tim Peeters
 * - Cleaned up the whole class. This class is now a subclass of
 *   bmia::vtkImageOrthogonalSlicesActor.
 *
 * 2006-12-26	Tim Peeters
 * - Use the new&improved TensorColoring.h
 *
 * 2007-04-03	Tim Peeters
 * - Stop using vtkDTILookupTables. Each dataset has its own
 *   lookup table now.
 * - Add UpdateInput().
 */

#ifndef bmia_vtkDTIPlanesActor_h
#define bmia_vtkDTIPlanesActor_h

#include "vtkImageOrthogonalSlicesActor.h"
#include "TensorColoring.h"

class vtkImageData;
class vtkLookupTable;

namespace bmia {

//class vtkDTILookupTables;
class vtkDTIDataManager;
class vtkMEVColoringFilter;

/**
 * Prop3D assembly that contains 3 orthogonal slices of a DTI dataset.
 */
class vtkDTIPlanesActor : public vtkImageOrthogonalSlicesActor
{
public:
  static vtkDTIPlanesActor* New();

  void SetDataManager(vtkDTIDataManager* manager);

//  void SetDTILookupTables(vtkDTILookupTables* luts);
//  vtkGetObjectMacro(DTILookupTables, vtkDTILookupTables);

  void SetColoringMeasure(int measure);
  vtkGetMacro(ColoringMeasure, int);

  void SetColoringType(int ctype);
  vtkGetMacro(ColoringType, int);
  void SetColoringTypeToLookupTable()
    { this->SetColoringType(TensorColoring::TypeLUT); };
  void SetColoringTypeToMEVToRGB()
    { this->SetColoringType(TensorColoring::TypeMEV); };
  void SetColoringTypeToMEVToWeightedRGB()
    { this->SetColoringType(TensorColoring::TypeWeightedMEV); };

  virtual void UpdateInput();

  vtkMEVColoringFilter* getMEVfilter() 
  { 
	  return this->MEVColoringFilter; 
  };

protected:
  vtkDTIPlanesActor();
  ~vtkDTIPlanesActor();

  vtkLookupTable* LookupTable;
//  vtkDTILookupTables* DTILookupTables;

  int ColoringMeasure;
  int ColoringType;

  /**
   * Set the lookup table. Use this->LookupTable if
   * this->DTILookupTables == NULL. Otheriwse use the lookup table from
   * DTILookupTables depending on the value of this->Coloring.
   */
//  void UpdateCurrentLookupTable();
  void UpdateColoring();
  void UpdateColoringLUT();
  void UpdateColoringMEV();
  void UpdateColoringMEVWeighted();

private:
  vtkDTIDataManager* DataManager;

  /**
   * The filter that computes e.g. anisotropy indices or helix angle
   * from the eigen system.
   */
//  vtkImageAlgorithm* Filter;

  vtkMEVColoringFilter* MEVColoringFilter;

  vtkDTIPlanesActor(const vtkDTIPlanesActor&);  // Not implemented.
  void operator=(const vtkDTIPlanesActor&);  // Not implemented.

}; // class vtkDTIPlanesActor
} // namespace bmia

#endif // bmia_vtkDTIPlanesActor_h
