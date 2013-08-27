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

/*
 * vtkGeometryGlyphFromSHBuilder.h
 *
 * 2011-05-09	Evert van Aart
 * - First version. 
 * 
 * 2011-05-10	Evert van Aart
 * - Fixed a memory allocation bug.
 * 
 * 2012-03-12	Ralph Brecheisen
 * - Including malloc.h is now conditional. On Mac OSX this file
 *   does not exist and can be ignored.
 */
 

#ifndef bmia_GeometryGlyphsPlugin_vtkGeometryGlyphFromSHBuilder_h
#define bmia_GeometryGlyphsPlugin_vtkGeometryGlyphFromSHBuilder_h


/** Includes - VTK */

#include <vtkPolyDataAlgorithm.h>
#include <vtkInformation.h>
#include <vtkObjectFactory.h>
#include <vtkMath.h>

/** Includes - Custom Files */

#include "vtkGeometryGlyphBuilder.h"
#include "HARDI/sphereTesselator.h"
#include "HARDI/SphereTriangulator.h"
#include "HARDI/HARDIMath.h"
#include "HARDI/HARDITransformationManager.h"

/** Includes - C++ */

#include <vector>

#if !defined(__APPLE__)
#include <malloc.h>
#endif

namespace bmia {


/** Class for building geometry glyphs from spherical harmonics. The parameters
	are largely similar to those of "vtkGeometryGlyphBuilder", which is the base
	class of this filter. The only two functions that differ are "computeGeometry",
	which constructs a tessellated sphere, and "Execute", which uses this sphere
	in combination with the SH coefficients to construct the glyphs. SH data of
	up to the eight order is supported. The input volume should have a scalar array
	with 1, 6, 15, 28 or 25 components.
*/

class vtkGeometryGlyphFromSHBuilder : protected vtkGeometryGlyphBuilder
{
	public:

		/** Constructor Call */

		static vtkGeometryGlyphFromSHBuilder * New();

		/** VTK Macro */

		vtkTypeMacro(vtkGeometryGlyphFromSHBuilder, vtkPolyDataAlgorithm);

		/** Compute the geometry of the glyphs. In this case, this is done by 
			tessellating a sphere with the specified tessellation order. The
			points of this unit sphere are stored in the "unitVectors" array, and
			their angles in spherical coordinates are stored in the "anglesArray"
			vector. The topology of the constructed sphere (i.e., it triangles)
			is stored in the "trianglesArray" array. 
			@param tessOrder	Tessellation order. */

		virtual bool computeGeometry(int tessOrder = 3);

	protected:

		/** Constructor. */

		vtkGeometryGlyphFromSHBuilder();

		/** Destructor. */

		~vtkGeometryGlyphFromSHBuilder();

		/** Execute the filter. */

		virtual void Execute();

	private:

		/** Array containing the angles, in spherical coordinates, of the points
			of the tessellated sphere. Note that, while the parent class uses
			zenith-azimuth spherical coordinates, this class uses 'regular'
			spherical coordinates, the difference being that the first angle
			is the elevation from the XY plane, rather than the angle from the
			positive Z-axis. This was done because the HARDI math functions
			expect spherical coordinates to use this system. */

		std::vector<double *> anglesArray;

		/** Compute the minimum and maximum radius for one glyph. If the normalization
			method is set to "None", the range 0-1 will be returned.
			@param radii	List of radii for one vector. 
			@param rMin		Output minimum radius.
			@param rMax		Output maximum radius. */

		void computeMinMaxRadii(std::vector<double> * radii, double & rMin, double & rMax);

}; // vtkGeometryGlyphFromSHBuilder


} // namespace bmia


#endif // bmia_GeometryGlyphsPlugin_vtkGeometryGlyphFromSHBuilder_h