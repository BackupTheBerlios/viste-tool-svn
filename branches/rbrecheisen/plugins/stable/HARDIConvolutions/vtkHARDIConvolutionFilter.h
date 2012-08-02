/*
 * vtkHARDIConvolutionFilter.h
 *
 * 2011-08-01	Evert van Aart
 * - First version
 *
 */


#ifndef bmia_HARDIConvolutionsPlugin_vtkHARDIConvolutionFilter_h
#define bmia_HARDIConvolutionsPlugin_vtkHARDIConvolutionFilter_h


/** Includes - VTK */

#include <vtkSimpleImageToImageFilter.h>
#include <vtkObjectFactory.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>

/** Includes - Qt */

#include <QStringList>

/** Includes - Custom Files */

#include "KernelNIfTIReader.h"
#include "KernelGenerator.h"


namespace bmia {


class vtkHARDIConvolutionFilter : public vtkSimpleImageToImageFilter
{
	public:

		static vtkHARDIConvolutionFilter * New();

		virtual void SimpleExecute(vtkImageData * input, vtkImageData * output);

		void setGenerator(KernelGenerator * rGen)
		{
			generator = rGen;
			niftiFileNames = NULL;
		}

		void setNIfTIFileNames(QStringList * rNames)
		{
			niftiFileNames = rNames;
			generator = NULL;
		}

		void setThreshold(double rT, bool rRelative)
		{
			Threshold = rT;
			useRelativethreshold = rRelative;
		}

		void setNumberOfIterations(int rIter)
		{
			numberOfIterations = rIter;
		}

	protected:

		vtkHARDIConvolutionFilter();

		~vtkHARDIConvolutionFilter();

		struct MapInfo
		{
			vtkIdType pointId;
			unsigned short dirId;
		};

		KernelGenerator * generator;

		QStringList * niftiFileNames;

		vtkIdType * neighborhood;
		double * kernelValues;
		MapInfo * kernelMask;
		int maskSize;

		vtkImageData * inputImage;
		vtkDoubleArray * imageValues;

		double Threshold;
		bool useRelativethreshold;

		int numberOfIterations;

	private:

		void getPositionNeighbourhood(int * ijkBase, int * kernelDim, int * imageDim);

		void maskKernel(int kernelSize, int numberOfDirections);

}; // class vtkHARDIConvolutionFilter

} // namespace


#endif // bmia_HARDIConvolutionsPlugin_vtkHARDIConvolutionFilter_h
