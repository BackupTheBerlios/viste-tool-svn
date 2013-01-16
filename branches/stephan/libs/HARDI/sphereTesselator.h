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

/***************************************************************************
 *   Copyright (C) 2004 by Christophe Lenglet                              *
 *   clenglet@sophia.inria.fr                                              *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#ifndef SPHERETESSELATOR_HPP
#define SPHERETESSELATOR_HPP

#include "tesselation.h"
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkPolyData.h>
#include <vtkPointLocator.h>
#include <vtkCleanPolyData.h>
//#include "defs.h"

namespace Visualization {

/*! http://davidf.faricy.net/polyhedra/Platonic_Solids.html */
	enum p_solid { cube,          // 0
				   dodecahedron,  // 1 
				   icosahedron,   // 2
				   octahedron,    // 3
				   tetrahedron }; // 4

/*! Perform tesselation of the unit sphere from a given platonic solid */
template<typename T>
class sphereTesselator {
public:

    sphereTesselator();
    sphereTesselator(const p_solid& ip);
    //sphereTesselator(const sphereTesselator<T>& st);

    virtual ~sphereTesselator();

    sphereTesselator<T>& operator= (const sphereTesselator<T>& st);

    void enableCClockWise(const bool& flag);
    void enableVerbosity(const bool& flag);
    void tesselate(const int& l);
    tesselation<T> getTesselation();
    void getvtkTesselation(const bool& clean, vtkPolyData* t);
	vtkPolyData * deformator (vtkPolyData * poly, double tensor[9]);

private:
    p_solid         m_initPolyhedra;
    tesselation<T>  m_tesselation;
    bool            m_cclockwise;

    vtkCleanPolyData* m_pDataCleaner;

    void m_initializeTesselation(const p_solid& ip);
    void m_flipOrientation(tesselation<T>* t);
    vertex<T> m_normalize(const vertex<T>& v);
    vertex<T> m_middle(const vertex<T>& v1, const vertex<T>& v2);
    bool m_verbose;
};
}
#endif

