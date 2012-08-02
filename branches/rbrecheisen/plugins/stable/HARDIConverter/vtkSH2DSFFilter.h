/*
 * vtkSH2DSFFilter.h
 *
 * 2011-08-05	Evert van Aart
 * - First version.
 *
 */


#ifndef bmia_HARDIConverterPlugin_vtkSH2DSFFilter_h
#define bmia_HARDIConverterPlugin_vtkSH2DSFFilter_h


/** Includes - VTK */

#include <vtkSimpleImageToImageFilter.h>
#include <vtkObjectFactory.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkPolyData.h>
#include <vtkPoints.h>
#include <vtkMath.h>
#include <vtkIntArray.h>
#include <vtkDoubleArray.h>

/** Includes - Custom Files */

#include "HARDI/sphereTesselator.h"
#include "HARDI/SphereTriangulator.h"
#include "HARDI/HARDITransformationManager.h"


namespace bmia {


/** This filter converts a volume containing Spherical Harmonics data to one 
	containing Discrete Sphere Functions (DSF). The DSF image has three arrays:
	One array containing the spherical angles, one containing the values per 
	voxel and angle (i.e., the radius for that angle), and one containing the
	triangulation of the computed glyphs. 
*/

class vtkSH2DSFFilter : public vtkSimpleImageToImageFilter
{
	public:

		/** Constructor Call */

		static vtkSH2DSFFilter * New();

		/** VTK Macro */

		vtkTypeMacro(vtkSH2DSFFilter, vtkSimpleImageToImageFilter);

		/** Execute the filter.
			@param input	Input SH image.
			@param output	Output DSF image. */

		virtual void SimpleExecute(vtkImageData * input, vtkImageData * output);

		/** Set the order of tessellation.
			@param rOder	Desired tessellation order. */

		void setTessOrder(int rOrder)
		{
			tessOrder = rOrder;
		}

	protected:

		/** Constructor. */

		vtkSH2DSFFilter();

		/** Destructor. */

		~vtkSH2DSFFilter();

	private:

		/** Tessellation order. */

		int tessOrder;


}; // vtkSH2DSFFilter


} // namespace bmia


#endif // bmia_HARDIConverterPlugin_vtkSH2DSFFilter_h