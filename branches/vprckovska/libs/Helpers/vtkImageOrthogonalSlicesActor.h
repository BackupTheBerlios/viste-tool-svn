/**
 * vtkImageOrthogonalSlicesActor.h
 * by Tim Peeters
 *
 * 2006-05-03	Tim Peeters
 * - First version
 */

#ifndef bmia_vtkImageOrthogonalSlicesActor_h
#define bmia_vtkImageOrthogonalSlicesActor_h

class vtkAlgorithmOutput;
class vtkImageData;
//class vtkLookupTable;
class vtkScalarsToColors;

#include <vtkAssembly.h>

namespace bmia {

class vtkImageSliceActor;

/**
 * Convenience class for showing 3 orthogonal slices of a volume
 * using three bmia::vtkImageSliceActors.
 */
class vtkImageOrthogonalSlicesActor : public vtkAssembly
{
public:
  static vtkImageOrthogonalSlicesActor* New();

  // Description:
  // Set/Get the input image to the actor.
  virtual void SetInput(vtkImageData *in);
  virtual vtkImageData *GetInput();
  //virtual void SetInputConnection(vtkAlgorithmOutput* input);

  // Description:
  // Set/Get the lookup table to map the data through.
  virtual void SetLookupTable(vtkScalarsToColors* lut);
  virtual vtkScalarsToColors* GetLookupTable();

  // Description:
  // Turn on/off the mapping of color scalars through the lookup table.
  // The default is Off. If Off, unsigned char scalars will be used
  // directly as texture. If On, scalars will be mapped through the
  // lookup table to generate 4-component unsigned char scalars.
  // This ivar does not affect other scalars like unsigned short, float,
  // etc. These scalars are always mapped through lookup tables.
  virtual int GetMapColorScalarsThroughLookupTable();
  virtual void SetMapColorScalarsThroughLookupTable(int map);
  virtual void MapColorScalarsThroughLookupTableOn()
    { this->SetMapColorScalarsThroughLookupTable(1); };
  virtual void MapColorScalarsThroughLookupTableOff()
    { this->SetMapColorScalarsThroughLookupTable(0); };

  // Description:
  // Turn on/off linear interpolation of the texture map when rendering.
  virtual int GetInterpolate();
  virtual void SetInterpolate(int interpolate);
  virtual void InterpolateOn()
    { this->SetInterpolate(1); };
  virtual void InterpolateOff()
    { this->SetInterpolate(0); };

  // Description:
  // Set/Get the X/Y/Z values for the YZ/XZ/XY slices.
  virtual void SetX(int x);
  virtual void SetY(int y);
  virtual void SetZ(int z);
  virtual int GetX();
  virtual int GetY();
  virtual int GetZ();

  // Description:
  // Center the slices in X/Y/Z directions.
  virtual void CenterSlices();

  // Description:
  // Return the minimum and maximum slice values.
  virtual int GetXMin();
  virtual int GetYMin();
  virtual int GetZMin();
  virtual int GetXMax();
  virtual int GetYMax();
  virtual int GetZMax();
  virtual void GetXRange(int range[2])
    { this->GetXRange(range[0], range[1]); };
  virtual void GetYRange(int range[2])
    { this->GetYRange(range[0], range[1]); };
  virtual void GetZRange(int range[2])
    { this->GetZRange(range[0], range[1]); };
  virtual void GetXRange(int &min, int &max);
  virtual void GetYRange(int &min, int &max);
  virtual void GetZRange(int &min, int &max);
  virtual int* GetXRange();
  virtual int* GetYRange();
  virtual int* GetZRange();

  // Description:
  // Return the actors used for each of the slices.
  virtual vtkImageSliceActor* GetSliceActor(int axis);
  virtual vtkImageSliceActor* GetXYActor()
    { return this->GetSliceActor(2); };
  virtual vtkImageSliceActor* GetXZActor()
    { return this->GetSliceActor(1); };
  virtual vtkImageSliceActor* GetYZActor()
    { return this->GetSliceActor(0); };

  // Description:
  // Call UpdateInput() for each of the 3 slice actors.
  virtual void UpdateInput();

  // Description:
  // Make any of the 3 slices (in)visible.
  virtual void SetSliceVisible(int axis, bool visible);
  virtual bool GetSliceVisible(int axis);
  virtual void SetXYSliceVisible(bool visible)
    { this->SetSliceVisible(2, visible); };
  virtual void SetXZSliceVisible(bool visible)
    { this->SetSliceVisible(1, visible); };
  virtual void SetYZSliceVisible(bool visible)
    { this->SetSliceVisible(0, visible); };
  virtual bool GetXYSliceVisible()
    { return this->GetSliceVisible(2); };
  virtual bool GetXZSliceVisible()
    { return this->GetSliceVisible(1); };
  virtual bool GetYZSliceVisible()
    { return this->GetSliceVisible(0); };

  virtual void SetVisibility(int show);

protected:
  vtkImageOrthogonalSlicesActor();
  ~vtkImageOrthogonalSlicesActor();

  vtkImageSliceActor* ImageSliceActors[3];

  bool IsSliceVisible[3];

private:
  vtkImageOrthogonalSlicesActor(const vtkImageOrthogonalSlicesActor&);  // Not implemented.
  void operator=(const vtkImageOrthogonalSlicesActor&);  // Not implemented.
}; // class vtkImageOrthogonalSlicesActor
} // namespace bmia
#endif // bmia_vtkImageOrthogonalSlicesActor_h
