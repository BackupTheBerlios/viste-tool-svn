/*
 * vtkStreamlineToHyperStreamline.h
 *
 * 2005-10-18	Anna Vilanova
 * - First version for the DTITool2 based on the class CStreamlineToStreamGeneralCylinder of the DTITool.
 *
 * 2010-09-24	Evert van Aart
 * - First version for the DTITool3. Added some minor code revisions, extended comments. 
 * - Class now uses pre-computed eigensystem data (eigenvectors and eigenvalues), rather than computing
 *	 them on-the-fly using the original DTI tensors.
 *
 */


#ifndef bmia__vtkStreamlineToHyperStreamline_H
#define bmia__vtkStreamlineToHyperStreamline_H


/** Includes - Main Header */

#include "DTITool.h"

/** Includes - Custom Files */

#include "vtkStreamlineToStreamGeneralCylinder.h"

/** Includes - VTK */

#include "vtkImageData.h"
#include "vtkDataArray.h"
#include "vtkObjectFactory.h"
#include "vtkCell.h"
#include "vtkPointData.h"
#include "vtkMath.h"


namespace bmia {


/** This class converts a streamline, which has been generated using one of the
	fiber tracking methods, into a hyper streamline, which is a 3D tube centered
	around the original streamline. Like the "vtkStreamlineToStreamTube" filter,
	it inherits from "vtkStreamlineToStreamGeneralCylinder", but the main difference
	is in the way it constructs the coordinate frames at each point along the fiber.
	Unlike the streamtubes class, this class uses image data (specifically, the
	eigenvectors and -values at each fiber point) to construct the frame and
	compute the radius of the tube. The resulting 3D shape thus directly shows
	the underlying DTI data. */


class vtkStreamlineToHyperStreamline : public vtkStreamlineToStreamGeneralCylinder
{
	public:
		
		/** Constructor Call */
		static vtkStreamlineToHyperStreamline * New();

		/** Set the hyper scale factor. */
		
		vtkSetClampMacro(HyperScale, float, 0.00001f, VTK_LARGE_FLOAT);
	
		/** Get the hyper scale factor. */
	
		vtkGetMacro(HyperScale, float);

		/** Set the image data containing the eigenvectors and -values.
			@param rEigenData	Input image data pointer. */
		
		void SetEigenData(vtkImageData * rEigenData)
		{
			// Check if the new pointer is the same as the old one
			if (this->eigenData != rEigenData)
			{
				/* Set the input of the filter. */
				this->vtkProcessObject::SetNthInput(1, rEigenData);

				/*Store the pointer. */
				this->eigenData = rEigenData;
			}
		}

		/** Get the image data containing the eigenvectors and -values. */
		
		vtkImageData * GetEigenData ()
		{
			return this->eigenData;
		}

		/** Structure containing the eigenvector and -values of one point. */
	
		typedef struct
		{
			double eigenVector0[3];
			double eigenVector1[3];
			double eigenVector2[3];
			double eigenValues[3];
		} eigenSystem;


	protected:
	
		/** Constructor */
	
		vtkStreamlineToHyperStreamline();
	
		/** Destructor */
	
		~vtkStreamlineToHyperStreamline() {}

		/** Eigensystems of the eight point in the current cell. */

		eigenSystem cellEigenSystems[8];

		/** Eigensystem interpolated at the current fiber position. */
	
		eigenSystem interpolatedEigenSystem;

		/** The maximum value of the second eigenvalue. When computing the radii of
			the tube, the eigenvalues at the current point are divided by this 
			maximum value. In points with second eigenvalue (almost) equal to the 
			maximum, the radius will be "1.0 * HyperScale". This should make the 
			behaviour of the hyperstreamline independent of the tensor scaling. */

		double maxEigenValue1;

		/** Image data containing the eigensystem for all points. */

		vtkImageData * eigenData;

		/** Data arrays of the eigenvectors and -values. */
	
		vtkFloatArray * eigenVectors0;
		vtkFloatArray * eigenVectors1;
		vtkFloatArray * eigenVectors2;
		vtkFloatArray * eigenValues0;
		vtkFloatArray * eigenValues1;
		vtkFloatArray * eigenValues2;

		/** The hyper scale factor, used to scale the radius of the tube. */
	
		float HyperScale;

		/** Tolerance variable, used in the "FindCell" function. */
	
		double tolerance;

		/** Update the output given the current input. */
		
		void Execute();

		/** Get the eigensystem at the input point. This function first finds the 
			cell containing the point, and subsequently calls the interpolation
			function. The result is stored in "interpolatedEigenSystem".
			@param point		3D point coordinates. */
	
		bool getEigenSystem(double * point);
	
		/** Linear interpolation of the eigensystem.
			@param cell			3D cell containing current point.
			@param w			Weights of the interpolation. */

		void interpolateEigenSystem(vtkCell * cell, double * w);

		/** Calculates the frame that is used to generate the points around
			the streamline. In this class, the frame is equal to the set of
			eigenvectors at the current point. This function also computes the
			radii of the tube, which depend on the second and third eigenvalues/
			@param prevFrame	Previous coordinate frame.
			@param newFrame		Compute coordinated frame.
			@param currentPoint	Current fiber point.
			@param prevPoint	Previous fiber point.
			@param lineSegment	Vector between previous and current point.
			@param pointId		ID of current input point. */

		virtual void calculateCoordinateFramePlane(		double prevFrame[3][3], 
														double newFrame[3][3], 
														double * currentPoint,
														double * prevPoint, 
														double * lineSegment, 
														vtkIdType pointId			);

		/** Calculates the first frame that is used to generate the points around
			the streamline. Practically the same as the previous function.
			@param prevFrame	Previous coordinate frame.
			@param newFrame		Compute coordinated frame.
			@param currentPoint	Current fiber point.
			@param prevPoint	Previous fiber point.
			@param lineSegment	Vector between previous and current point.
			@param pointId		ID of current input point. */

		virtual void calculateFirstCoordinateFramePlane(double prevFrame[3][3], 
														double newFrame[3][3], 
														double * currentPoint,
														double * prevPoint, 
														double * lineSegment, 
														vtkIdType pointId			);

		/** Ensures 1) that the Z-vector of the frame is aligned with the vector
			between the last two fiber points (i.e., their dot product is positive),
			and 2) that the frame is right-handed. 
			@param newFrame		Current coordinate frame.
			@param lineSegment	Vector between last two fiber points. */

		void fixFirstVectors(double newFrame[3][3], double * lineSegment);

		/** First call "fixFirstVectors"; subsequently also checks if the X-vector
			of the new frame is aligned with that of the old one. If not, flip both
			the X- and the Y-vectors (to maintain right-handedness) of the new frame.
			@param prevFrame	Previous coordinate frame.
			@param newFrame		Current coordinate frame.
			@param lineSegment	Vector between last two fiber points. */
	
		void fixVectors(double prevFrame[3][3], double newFrame[3][3], double * lineSegment);

}; // class vtkStreamlineToHyperStreamline


} // namespace bmia


#endif // bmia__vtkStreamlineToHyperStreamline_H
