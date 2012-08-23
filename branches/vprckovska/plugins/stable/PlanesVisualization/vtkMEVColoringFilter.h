/*
 * vtkMEVColoringFilter.h
 *
 * 2006-04-13	Tim Peeters
 * - First version
 *
 * 2006-08-03	Tim Peeters
 * - Fixed bug. It now finally works with
 *     vtkPointData * weightingPD = this->WeightingVolume->GetPointData();
 *   instead of
 *     vtkPointData * weightingPD = input->GetPointData();
 * - Call "this->WeightingVolume->Update()" before it is used.
 *
 *  2007-10-17	Tim Peeters
 *  - Added "ShiftValues".
 *
 *  2007-10-19	Tim Peeters
 *  - Call "SetNumberOfScalarComponents(3)" on output. Some filters check
 *    for this value for the input, and it was not set correctly.
 *
 * 2011-01-14	Evert van Aart
 * - Fixed a huge memory leak. The 3-component vector ("in_vector") was allocated using "new"
 *   for every single point, but never deleted. Replaced it with a static array, because there
 *   was no reason to have "in_vector" be dynamically allocated.
 *
 * 2011-04-14	Evert van Aart
 * - Weight values are now automatically normalized to the range 0-1.
 * - Cleaned up code, added comments. 
 *
 * 2011-07-12	Evert van Aart
 * - Removed the warning about "SetScalarType". I'm still not exactly
 *   sure why it cause problems, but we really do not need to call it, so I simply
 *   removed it altogether.
 *
 */


#ifndef bmia_vtkMEVColoringFilter_h
#define bmia_vtkMEVColoringFilter_h


/** Includes - VTK */

#include <vtkSimpleImageToImageFilter.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkUnsignedCharArray.h>
#include <vtkObjectFactory.h>
#include <vtkPointData.h>


namespace bmia {

/** This class filters a volume of eigenvectors to a volume of colors by mapping
	the XYZ-components of the vectors to RGB-components of the color. An optional
	scalar volume can be used as a weighting for the colors. Colors have unsigned 
	char type so that they can be used on textures directly.
*/

class vtkMEVColoringFilter : public vtkSimpleImageToImageFilter
{
	public:
	
		/** Index of the X-component (0, 1, or 2). */

		static int xIndexRGB;
		
		/** Index of the Y-component (0, 1, or 2). */
		
		static int yIndexRGB;
		
		/** Index of the Z-component (0, 1, or 2). */
		
		static int zIndexRGB;

		/** Constructor Call */

		static vtkMEVColoringFilter * New();

		/** Set the scalar volume that is used for the weighting. If the weighting 
			volume is set to NULL, no weighting is applied.
			@param w	Input weighting volume. */
  
		void SetWeightingVolume(vtkImageData * w);

		/** Returns the current weighting volume. */
  
		vtkGetObjectMacro(WeightingVolume, vtkImageData);

		/** Turn "ShiftValues" on or off. If "ShiftValues" is OFF, in the output 
			the RGB components will be the absolute values of the XYZ components 
			of the input. If "ShiftValues" is ON, then no absolute will be taken 
			but the value range will be shifted from -1...1 to 0...1 (before 
			converting to 0...255) by dividing the value by 2 and adding 0.5.
			Default value for "ShiftValues" is OFF. */
  
		vtkSetMacro(ShiftValues, bool);

		/** Turn "ShiftValues" on or off. */
  
		vtkBooleanMacro(ShiftValues, bool);

		/** Return the current setting for "ShiftValues. */

		vtkGetMacro(ShiftValues, bool);

	protected:

		/** Constructor */

		vtkMEVColoringFilter();

		/** Destructor */

		~vtkMEVColoringFilter();

		/** This function is called when the filter must be executed. Input is 
			the eigenvector volume, output is the RGB volume. 
			@param input		Volume containing eigensystem. 
			@param output		Scalar volume containing RGB values. */

		virtual void SimpleExecute(vtkImageData * input, vtkImageData * output);

	private:
  
		/** The (optional) volume used to weight the RGB colors. */
  
		vtkImageData * WeightingVolume;

		/** Whether or not the colors should be shifted. */

		bool ShiftValues;

}; // class vtkMEVColoringFilter


} // namespace bmia


#endif // bmia_vtkMEVColoringFilter_h
