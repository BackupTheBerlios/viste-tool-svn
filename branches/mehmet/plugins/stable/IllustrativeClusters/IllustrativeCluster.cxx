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

/** Includes */

#include "IllustrativeCluster.h"
#include "vtkIllustrativeFiberBundleMapper.h"


namespace bmia {


//-----------------------------[ Constructor ]-----------------------------\\

IllustrativeCluster::IllustrativeCluster(const QString & name, 
										 vtkSmartPointer<vtkActor> actor, 
										 vtkSmartPointer<vtkIllustrativeFiberBundleMapper> mapper)
										 : mName(name), mActor(actor), mMapper(mapper)
{
	// Default configuration settings
	this->mConfiguration.lineColor = QColor(156, 110, 110);
	this->mConfiguration.fillColor = QColor(240, 209, 209);

	this->mConfiguration.haloWidth = 0.4f;
	this->mConfiguration.haloDepth = 0.1f;

	this->mConfiguration.enableLighting = true;
	this->mConfiguration.phongConstants = Vector3(0.0, 0.7, 0.2);
	this->mConfiguration.specularPower = 1;
	this->mConfiguration.minLuminosity = 0;
	this->mConfiguration.maxLuminosity = 1;

	this->mConfiguration.enableCurvatureStrokes = false;
	this->mConfiguration.minStrokeWidth = 0.05f;
	this->mConfiguration.maxStrokeWidth = 0.2f;

	this->mConfiguration.enableSilhouette = true;
	this->mConfiguration.silhouetteWidth = 3;
	this->mConfiguration.contourWidth = 2;
	this->mConfiguration.depthThreshold = 10.0f;
}


//------------------------------[ Destructor ]-----------------------------\\

IllustrativeCluster::~IllustrativeCluster()
{
	// Don't do anything. The plugin itself removes the actor from the assembly,
	// which reduces its reference count to zero, and which in turn deleted
	// both the actor and its mapper.
}


//------------------------------[ SetColors ]------------------------------\\

void IllustrativeCluster::SetColors(QString lineColor, QString fillColor)
{
	// Set the new colors
	this->mConfiguration.lineColor.setNamedColor(lineColor);
	this->mConfiguration.fillColor.setNamedColor(fillColor);
}


} // namespace bmia
