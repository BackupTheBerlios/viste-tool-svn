#ifndef __vtkImageDataToDistanceTransform_h
#define __vtkImageDataToDistanceTransform_h

// Includes VTK
#include <vtkObject.h>
#include <vtkImageData.h>

// Includes STL
#include <vector>

/** @class	vtkImageDataToDistanceTransform
	@brief	Computes distance transform from the input image data
	*/
class vtkImageDataToDistanceTransform : public vtkObject
{
public:

	/** Creates new instance */
	static vtkImageDataToDistanceTransform * New();
	vtkTypeRevisionMacro( vtkImageDataToDistanceTransform, vtkObject );

	/** Sets/gets the input volume. The volume should be a scalar volume
		consisting of gray values */
	void SetInput( vtkImageData * data );
	vtkImageData * GetInput();

	/** Sets/gets the scalar threshold to be applied to the input volume gray
		values before the distance transform is executed
		@param threshold	The gray value threshold */
	void  SetThreshold( float threshold );
	float GetThreshold();

    /** Inverts distances such that points at greater distance are assigned
        a lower value, e.g., for distance D, the value becomes 1 / D. */
    void SetDistanceInverted( bool inverted );
    bool IsDistanceInverted();

	/** Uploads the image data */
	void Upload();

	/** Executes filter */
	void Execute();

	/** Downloads the resulting distance transform */
	void Download();

	/** Returns distance transform of input volume as VTK image data */
	vtkImageData * GetOutput();

	/** Returns output voronoi containing for each voxel the position
		of the closest non-empty voxel */
	vtkImageData * GetOutputVoronoi();

protected:

	/** Constructor/destructor */
	vtkImageDataToDistanceTransform();
	virtual ~vtkImageDataToDistanceTransform();

private:

	/** Computes next power of two for given number
		@param number the number */
	int NextPowerOfTwo( int number );

	/** Converts unsigned char voxels to list of 3D vertices. Voxels are
		thresholded with the member variable Threshold. The number of
		vertices found is returned
		@param voxels The voxel data
		@param dimX Dimension X
		@param dimY Dimension Y
		@param dimZ Dimension Z
		@return List of vertices */
	std::vector< float * > * UnsignedCharVoxelsToVertices( unsigned char * voxels, int dimX, int dimY, int dimZ );

	/** Converts unsigned short voxels to list of 3D vertices. Voxels are
		thresholded with the member variable Threshold. The number of
		vertices found is returned
		@param voxels The voxel data
		@param dimX Dimension X
		@param dimY Dimension Y
		@param dimZ Dimension Z
		@return List of vertices */
	std::vector< float * > * UnsignedShortVoxelsToVertices( unsigned short * voxels, int dimX, int dimY, int dimZ );

	/** Converts floating-point voxels to list of 3D vertices. Voxels are
		thresholded with the member variable Threshold. The number of
		vertices found is returned
		@param voxels The voxel data
		@param dimX Dimension X
		@param dimY Dimension Y
		@param dimZ Dimension Z
		@return List of vertices */
	std::vector< float * > * FloatVoxelsToVertices( float * voxels, int dimX, int dimY, int dimZ );

	vtkImageData	* Data;						// Image data on which the DT is computed
	vtkImageData	* DistanceTransform;		// Computed distance transform
	vtkImageData	* Voronoi;					// Computed voronoi
	float Threshold;							// Threshold for gray values
    bool DistanceInverted;                      // Whether distances are inverted or not
	int * InputVoronoi;							// The input Voronoi vertices for the DT
};

#endif
