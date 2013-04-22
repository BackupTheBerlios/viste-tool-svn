/**
 * Copyright (c) 2012, Biomedical Image Analysis Eindhoven (BMIA/e)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 *   - Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 * 
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the 
 *     distribution.
 * 
 *   - Neither the name of Eindhoven University of Technology nor the
 *     names of its contributors may be used to endorse or promote 
 *     products derived from this software without specific prior 
 *     written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

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
