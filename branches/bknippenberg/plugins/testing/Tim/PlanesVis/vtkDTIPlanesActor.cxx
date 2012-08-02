/**
 * vtkDTIPlanesActor.cxx
 * by Tim Peeters
 *
 * 2005-11-14	Tim Peeters
 * - First version
 *
 * 2006-01-23	Tim Peeters
 * - Added SetColoring() function.
 *
 * 2006-01-24	Tim Peeters
 * - Added SetInterpolate(int) function.
 *
 * 2006-03-28	Tim Peeters
 * - Added vtkLookupTable stuff.
 *
 * 2006-04-10	Tim Peeters
 * - Added SetLookupTable(vtkLookupTable* lut) function.
 * - Removed bmia::vtkTensorsToColors stuff.
 *
 * 2006-04-11	Tim Peeters
 * - Added SetDTILookupTables().
 * - Compute eigen systems and anisotropy indices here. Pass the
 *   anisotropy indices to the slice actors.
 *
 * 2006-04-12	Tim Peeters
 * - Removed all filtering stuff. This is now in bmia::vtkDTIDataManager.
 * - Made input a vtkDTIDataManager instead of vtkImageData.
 *
 * 2006-04-13	Tim Peeters
 * - Added support for other color coding types besides mapping scalar
 *   indices through color lookup tables.
 *
 * 2006-04-26	Tim Peeters
 * - Moved defines to TensorColoring.h
 *
 * 2006-05-01	Tim Peeters
 * - Use input connections and output ports.
 *
 * 2006-05-03	Tim Peeters
 * - Cleaned up the whole class. Most functionality is now in
 *   bmia::vtkImageOrthogonalSlicesActor.
 *
 * 2006-12-26	Tim Peeters
 * - Simplified using ScalarMeasures.h
 *
 * 2007-04-03	Tim Peeters
 * - Stop using vtkDTILookupTables. Each dataset has its own LUT now.
 * - Add UpdateInput() to update the coloring and then call UpdateInput()
 *   of superclass.
 *
 * 2008-06-09   Adriaan Versteeg
 * - Added Test in function Set Coloring Measure to check if scalar data is available
 */

#include "vtkDTIPlanesActor.h"
#include <vtkImageData.h>
#include <vtkObjectFactory.h>
#include <vtkProperty.h>
#include <vtkLookupTable.h>
#include "vtkDTIDataManager.h"
#include "vtkMEVColoringFilter.h"
#include "TensorColoring.h"
#include "ScalarMeasures.h"
#include "vtkProperty.h"

namespace bmia {

vtkStandardNewMacro(vtkDTIPlanesActor);

vtkDTIPlanesActor::vtkDTIPlanesActor()
{
  this->DataManager = NULL;

  this->MEVColoringFilter = vtkMEVColoringFilter::New();

  this->ColoringMeasure = ScalarMeasures::FA;
  this->ColoringType = TensorColoring::TypeLUT;
}

vtkDTIPlanesActor::~vtkDTIPlanesActor()
{
  if (this->DataManager)
    {
    this->DataManager->UnRegister(this);
    this->DataManager = NULL;
    }
}

void vtkDTIPlanesActor::SetDataManager(vtkDTIDataManager* manager)
{
  if (this->DataManager == manager) return;
  if (this->DataManager) this->DataManager->UnRegister(this);
  this->DataManager = manager;
  if (this->DataManager)
    {
    this->DataManager->Register(this);
    this->MEVColoringFilter->SetInputConnection(this->DataManager->GetEigensystemOutputPort());
//    this->UpdateColoring();
    }

  this->Modified();
} 

void vtkDTIPlanesActor::UpdateColoring()
{
  vtkDebugMacro(<<"Updating coloring!");
  if (!this->DataManager)
    {
    vtkErrorMacro(<<"No data manager was specified!");
    return;
    }

  switch (this->ColoringType)
    {
    case TensorColoring::TypeMEV:
      this->UpdateColoringMEV();
      break;
    case TensorColoring::TypeWeightedMEV:
      this->UpdateColoringMEVWeighted();
      break;
    case TensorColoring::TypeLUT:
      this->UpdateColoringLUT();
      break;
    default:
      vtkErrorMacro(<<"Unknown or unsupported coloring type: "<<this->ColoringType);
    }
}

void vtkDTIPlanesActor::UpdateColoringMEV()
{
  vtkDebugMacro(<<"Updating coloring: MEV");
  this->MEVColoringFilter->SetWeightingVolume(NULL);
  this->SetInput(this->MEVColoringFilter->GetOutput());
  this->MapColorScalarsThroughLookupTableOff();
}

void vtkDTIPlanesActor::UpdateColoringMEVWeighted()
{
  vtkDebugMacro(<<"Updating coloring: Weighted MEV");
  this->MEVColoringFilter->SetWeightingVolume(this->DataManager->GetScalarOutput(this->ColoringMeasure));

  this->SetInputConnection(this->MEVColoringFilter->GetOutputPort());
  this->MapColorScalarsThroughLookupTableOff();
}

void vtkDTIPlanesActor::UpdateColoringLUT()
{
  vtkDebugMacro(<<"Updating Coloring: LUT");
  this->SetLookupTable(this->DataManager->GetLookupTable(this->ColoringMeasure));
  this->SetInputConnection(this->DataManager->GetScalarOutputPort(this->ColoringMeasure));
  this->MapColorScalarsThroughLookupTableOn();
  vtkDebugMacro(<<"Finished updating LUT coloring pipeline.");
}

void vtkDTIPlanesActor::SetColoringMeasure(int measure)
{
  if(DataManager->GetScalarOutput(measure)!=NULL)
  {
	this->ColoringMeasure = measure;
	this->UpdateColoring();
  }
}

void vtkDTIPlanesActor::SetColoringType(int ctype)
{
  this->ColoringType = ctype;
  this->UpdateColoring();
}

void vtkDTIPlanesActor::UpdateInput()
{
  if (!this->DataManager)
    {
    vtkErrorMacro(<<"No data manager!");
    return;
    }

  this->MEVColoringFilter->SetInputConnection(this->DataManager->GetEigensystemOutputPort());


  this->UpdateColoring();
  this->vtkImageOrthogonalSlicesActor::UpdateInput();
}

} // namespace bmia
