/*
 * KernelGenerator.h
 *
 * 2011-08-02	Evert van Aart
 * - First version. 
 * 
 */


#ifndef bmia_HARDIConvolutionsPlugin_KernelGenerator_h
#define bmia_HARDIConvolutionsPlugin_KernelGenerator_h


/** Includes - VTK */

#include <vtkObject.h>
#include <vtkMath.h>
#include <vtkObjectFactory.h>
#include <vtkMatrix4x4.h>

/** Includes - Qt */

#include <QStringList>
#include <QProgressDialog>

/** Includes - C++ */

#include <vector>

/** Includes - Custom Files */

#include "KernelNIfTIWriter.h"


namespace bmia {


/** Class used to generate kernel images. The generator can either create a whole
	group of kernel images, and write them to NIfTI files, or create a single 
	kernel image for one specific direction (which happens when it is directly 
	controlled by the convolution filter). The most important parameters are the
	size and the kernel type. Currently, only cubic kernel images with odd dimensions
	(e.g, 3x3x3, 5x5x5) are supported. The list of supported kernel types can be
	found in the "KernelType" enum. Each kernel type has a number of parameters.
	Additionally, the user can choose to normalize the kernel images, using L1
	normalization. 
*/

class KernelGenerator : public vtkObject
{
	public:

		/** Kernel type. Determines the way the kernel values are computed. */
		enum KernelType
		{
			KT_Duits = 0,		/**< Diffusion kernels of Remco Duits. */
			KT_Barmpoutis		/**< Barmpoutis kernels. */
		};

		/** Constructor Call */

		static KernelGenerator * New();

		/** Set the kernel type.
			@param rType	Desired kernel type. */

		void SetKernelType(KernelType rType)
		{
			Type = rType;
		}

		/** Set "D33", used by the Duits kernels. */

		vtkSetMacro(D33, double);

		/** Set "D44", used by the Duits kernels. */

		vtkSetMacro(D44, double);

		/** Set "t", used by the Duits kernels. */

		vtkSetMacro(T, double);

		/** Set "Sigma", used by the Barmpoutis kernels. */

		vtkSetMacro(Sigma, double);

		/** Set "Kappa", used by the Barmpoutis kernels. */

		vtkSetMacro(Kappa, double);

		/** Set whether or not to normalize the computed kernels. */

		vtkSetMacro(NormalizeKernels, bool);

		/** Set the extent of the kernel images. */

		vtkSetVector6Macro(Extent, int);

		/** Get the dimensions of the kernel (computed from the extents). */

		vtkGetVector3Macro(Dim, int);

		/** Set the spacing of the HARDI image. */

		void SetSpacing(double * rSpacing)
		{
			Spacing[0] = rSpacing[0];
			Spacing[1] = rSpacing[1];
			Spacing[2] = rSpacing[2];
			spacingHasBeenTransformed = false;
		}

		/** Set the transformation matrix. Used to determine the spacing of the
			HARDI image, see comments for "tM".
			@param rM			Input transformation matrix. */

		void SetTransformationMatrix(vtkMatrix4x4 * rM)
		{
			tM = rM;
		}

		/** Set the directions of the discrete sphere functions. All directions 
			should be stored as 3D unit vectors. 
			@param lst			Vector containing the directions. */

		void SetDirections(std::vector<double *> * lst)
		{
			directions = lst;
		}

		/** Return the list of file names of the generated NIfTI files. */

		QStringList * GetKernelImageFileNames()
		{
			return fileNameList;
		}

		/** Set the path and the base name of the output NIfTI files. 
			@param rPath		Path (ending with a slash). 
			@param rBaseName	Base file name. */
		
		void setFileNameAndPath(QString rPath, QString rBaseName)
		{
			absolutePath = rPath;
			baseName = rBaseName;
		}

		/** Build a family (group) of kernel images, one for each of the input
			directions. Each kernel image is written to a NIfTI file named
			"baseNameXX.nii", where "baseName" is the base name chosen by the
			user, and "XX" is an integer between zero and the number of 
			directions. Returns false if something went wrong (e.g., an
			error writing one of the NIfTI files. This method is called when
			the user clicks the "Write Kernels to NIfTI Files" button. */

		bool BuildKernelFamily();

		/** Build a single kernel image, for the specified directions. Called by
			the convolution filter if the user has chosen the kernel generator
			as the source for kernel images (i.e., the kernels generated directly
			when applying the convolutions). In the output array, the directions
			are the outermost dimension (i.e., all kernel values for one direction
			are grouped together, followed by all kernel values for the next 
			direction, and so on).
			@param firstDirectionID	Index of the main direction of the kernel image.
			@param outArray			Output array containing the kernel values. */

		void BuildSingleKernel(int firstDirectionID, double * outArray);

		/** Compute the dimensions from the extents. */

		void UpdateDimensions();

	protected:

		/** Constructor */

		KernelGenerator();

		/** Destructor */

		~KernelGenerator();

	private:

		/** Very small value, used to compare with zero. */

		static double CLOSE_TO_ZERO;

		/** Compute a single kernel value for one voxel of the kernel image, based 
			on the position of the voxel relative to the center, and two 3D unit
			vector representing the main direction of the kernel image and the
			direction of the current value, respectively. Based on these three
			inputs, a scalar value is computed using the algorithm by Remco Duits.
			@param ijk		Vector between kernel center and current voxel.
			@param d1		Main direction of kernel image.
			@param d2		Direction of current kernel value. */

		double ComputeKernelValueDuits(double * ijk, double * d1, double * d2);

		/** Compute a single kernel value for one voxel of the kernel image, based 
			on the position of the voxel relative to the center, and two 3D unit
			vector representing the main direction of the kernel image and the
			direction of the current value, respectively. Based on these three
			inputs, a scalar value is computed using the algorithm by Barmpoutis.
			@param ijk		Vector between kernel center and current voxel.
			@param d1		Main direction of kernel image.
			@param d2		Direction of current kernel value. */

		double ComputeKernelValueBarmpoutis(double * ijk, double * d1, double * d2);

		/** Used for the Duits kernels. The output vector is allocated in this function.
			@param n		Input vector, three elements.
			@param y		Input vector, three elements. */

		double * RTnyforCPP(double * n, double * y);

		/** Converts a 3D vector to two Euler angles. First normalizes the vector.
			Calls "ConvertToEuler", which also allocates the output vector.
			@param x		Input vector, three elements. */

		double * ConvertToEulerN(double * x);

		/** Converts a 3D vector to two Euler angles. Allocates the output vector.
			@param x		Input vector, three elements. */

		double * ConvertToEuler(double * x);

		/** Used for the Duits kernels. Called by "KernelSE3".
			@param D11		Always equal to "D44".
			@param D22		Always equal to "D33". 
			@param x		Half the "z" value of "KernelSE3". 
			@param y		Either "x" or "-y" of "KernelSE3".
			@param theta	One of the two Euler angles. */

		double KernelSE2(double D11, double D22, double x, double y, double theta);

		/** Used for the Duits kernels. Called by the other version of "KernelSE3".
			@param x		Equal to "x[0]" of the other "KernelSE3".
			@param y		Equal to "x[1]" of the other "KernelSE3".
			@param z		Equal to "x[2]" of the other "KernelSE3".
			@param beta		Equal to "r[0]" of the other "KernelSE3".
			@param gamma 	Equal to "r[1]" of the other "KernelSE3". */

		double KernelSE3(double x, double y, double z, double beta, double gamma);

		/** Computes a single value from a 3D vector and a pair of Euler angles.
			@param x		3D vector.
			@param r		Set of Euler angles. */

		double KernelSE3(double * x, double * r);

		/** Approximation of "(0.5 * ang) / tan(ang / 2.0)". Used by the "KernelSE2"
			function when high precision is not required.
			@param ang		Input angle. */

		inline static double approx(double ang);

		/** Returns "(0.5 * ang) / tan(ang / 2.0)". Used by "KernelSE2".
			@param ang		Input angle. */

		inline static double exact(double ang);

		/** Compute "K_distance", used for the Barmpoutis kernels.
			@param y		Vector between kernel center and current voxel. 
			@param sigma	Set by the user. */

		double calculateKDistance(double * y, double sigma);

		/** Compute "K_fiber", used for the Barmpoutis kernels.
			@param y		Vector between kernel center and current voxel. 
			@param d1		Main direction of kernel image.
			@param kappa	Set by the user. */

		double calculateKFiber(double * y, double * r, double kappa);

		/** Compute "K_orientation", used for the Barmpoutis kernels.
			@param r		Main direction of kernel image.
			@param v		Direction of current kernel value.
			@param kappa	Set by the user. */

		double calculateKOrientation(double * r, double * v, double kappa);

		//newcode
		void UpdateSpacing();

		KernelType	Type;				/**< Kernel type. */
		double		D33;				/**< Used for the Duits kernels. */
		double		D44;				/**< Used for the Duits kernels. */
		double		T;					/**< Used for the Duits kernels. */
		double		Sigma;				/**< Used for the Barmpoutis kernels. */
		double		Kappa;				/**< Used for the Barmpoutis kernels. */
		bool		NormalizeKernels;	/**< If true, kernels are normalized. */
		int			Extent[6];			/**< Extents of the kernel image(s). */
		int			Dim[3];				/**< Kernel dimensions, computed from extents. */
		double		Spacing[3];			/**< HARDI image spacing. */

		/** Transformation matrix, used to determine the actual spacing. Since the
			values of one kernel glyph depend on its position relative to the center
			of the kernel image, we should take the spacing into account when 
			computing these values. However, often the spacing is set to (1, 1, 1),
			and the transformation matrix is used to indirectly apply spacing, so
			if we want to accurately compute the kernel, we need to take this into
			account. */

		vtkMatrix4x4 * tM;

		/** True if the current spacing values have already been transformed
			(in the function "UpdateSpacing"). */

		bool spacingHasBeenTransformed;

		QString absolutePath;			/**< Absolute path for the NIfTI files. */
		QString baseName;				/**< Base name for the NIfTI files. */
		QStringList * fileNameList;		/**< File names of the generated NIfTI files. */

		/** Vector containing the direction of the discrete sphere function. 
			Directions are stored as 3D vectors of unit length. */

		std::vector<double *> * directions;

}; // class KernelGenerator


} // namespace bmia


#endif // bmia_HARDIConvolutionsPlugin_KernelGenerator_h