#ifndef bmia_IllustrativeClustersPlugin_IllustrativeCluster_h
#define bmia_IllustrativeClustersPlugin_IllustrativeCluster_h


/** Includes - Qt */

#include <QVector3D>
#include <QColor>
#include <QString>

/** Includes - VTK */

#include <vtkActor.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>

/** Includes - Custom Files */

#include "Math/Vector3.h"


using ICMath::Vector3;


namespace bmia {


class vtkIllustrativeFiberBundleMapper;


/** Class representing one cluster of fibers, for use with the "IllustrativeClustersPlugin".
	Mainly used for storing the settings of the cluster (configuration), and pointers to
	its mapper and actor. 
*/

class IllustrativeCluster
{	
	public:

		/** This struct contains all settings of the current cluster. */

		struct Configuration
		{
			QColor		lineColor;
			QColor		fillColor;

			float		haloWidth;
			float		haloDepth; 

			bool		enableCurvatureStrokes;
			float		minStrokeWidth;
			float		maxStrokeWidth;

			bool		enableLighting;
			Vector3		phongConstants;
			int			specularPower;
			float		minLuminosity;
			float		maxLuminosity;

			bool		enableSilhouette;
			int			silhouetteWidth;
			int			contourWidth;	
			float		depthThreshold;	
		};

		IllustrativeCluster(const QString & name, 
			vtkSmartPointer<vtkActor> actor, 
			vtkSmartPointer<vtkIllustrativeFiberBundleMapper> mapper);

		~IllustrativeCluster();

		QString getName() 
		{ 
			return mName; 
		}

		Configuration getConfiguration()
		{ 
			return mConfiguration; 
		}

		vtkActor * getActor()
		{ 
			return mActor; 
		}

		vtkIllustrativeFiberBundleMapper * getMapper()
		{ 
			return mMapper; 
		}

		void updateName(const QString name) 
		{ 
			mName = name; 
		}

		void updateConfiguration(const Configuration & configuration) 
		{ 
			mConfiguration = configuration; 
		}

		void SetColors(QString lineColor, QString fillColor);
		
	private:

		QString mName;
		Configuration mConfiguration;

		vtkSmartPointer<vtkActor> mActor;
		vtkSmartPointer<vtkIllustrativeFiberBundleMapper> mMapper;

}; // class IllustrativeCluster


} // namespace bmia


#endif // bmia_IllustrativeClustersPlugin_IllustrativeCluster_h
