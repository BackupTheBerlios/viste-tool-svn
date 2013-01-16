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
 * geodesicFiberTracker.h
 *
 * 2011-06-01	Evert van Aart
 * - First Version.
 *
 */


#ifndef bmia_FiberTrackingPlugin_geodesicFiberTracker_h
#define bmia_FiberTrackingPlugin_geodesicFiberTracker_h


/** Includes - Custom Files */

#include "streamlineTracker.h"
#include "geodesicPreProcessor.h"
#include "geodesicFiberNeighborhood.h"
#include "vtkFiberTrackingGeodesicFilter.h"


namespace bmia {


/** Tracker used to compute geodesic 'fibers'. Inherits from "streamlineTracker",
	but overwrites its core functionality (the "calculateFiber" function). Uses
	a "geodesicNeighborhood" object to keep track of the data around the current
	fiber position; this neighborhood class is also used to interpolate data and
	to compute the Christoffel symbols, which are used to solve the ODE integration
	step. Several different ODE solvers are supported.
*/

class geodesicFiberTracker : public streamlineTracker
{
	public:

		/** Constructor */

		geodesicFiberTracker();

		/** Destructor */

		~geodesicFiberTracker();

		/** Compute a single geodesic fiber. The input list initially contains
			a single point, containing the initial position and direction of the
			fibers. Computed points are added to this list.
			@param pointList	List of fiber points. */

		virtual void calculateFiber(std::vector<streamlinePoint> * pointList);

		/** Set the preprocessor, which is used to compute the input data of the
			algorithm (preprocessed tensors and their inverse) from the original
			DTI tensors. Created and destroyed in "vtkFiberTrackingGeodesicFilter".
			@param rPP			Pointer to preprocessor. */

		void setPreProcessor(geodesicPreProcessor * rPP)
		{
			this->pp = rPP;
		}

		/** Set the ODE solver used to compute the fiber.
			@param rSolver		Desired ODE solver. */

		void setSolver(vtkFiberTrackingGeodesicFilter::ODESolver rSolver)
		{
			mySolver = rSolver;
		}
				
		/** Enumeration for the Christoffel symbols. The first number is the superscript
			value of the symbol; the last two numbers are the two subscript values. */

		enum ChristoffelIndex
		{
			G111=0,	G112,	G122,	G113,	G123,	G133,	
			G211,	G212,	G222,	G213,	G223,	G233,	
			G311,	G312,	G322,	G313,	G323,	G333
		};


	private:

		/** Preprocessor, used to compute the input data of the algorithm 
			(preprocessed tensors and their inverse) from the original DTI tensors. */

		geodesicPreProcessor * pp;

		/** Fiber neighborhood, used to store relevant data (preprocessed tensors,
			inverted tensors, derivatives of the inverse tensors, scalar values,
			and point indices) for a 4x4x4 voxel neighborhood around the current
			fiber position. Used to get the Christoffel symbols and scalar values. */

		geodesicFiberNeighborhood * gfnh;

	protected:

		/** Solve a single integration step, using the position and direction of 
			the current point. The resulting position and direction are written to
			the "nextPoint" variable (inherited from "streamlineTracker").
			@param currentCell		Pointer to the current cell.
			@param currentCellId	Index of the current cell.
			@param weights			Integration weights. */

		virtual bool solveIntegrationStep(vtkCell * currentCell, vtkIdType currentCellId, double * weights);

	private:

		/** Compute the next position and direction using a simple Euler step.
			@param currentDelta		Current fiber direction.
			@param nextDelta		Next fiber direction.
			@param currentPosition	Current fiber position.
			@param nextPosition		Next fiber position.
			@param Csymbols			Christoffel symbols computed at "currentPosition. */

		void solverEuler(double * currentDelta, double * nextDelta, double * currentPosition, double * nextPosition, double * Csymbols);
	
		/** Compute the next position and direction using a second-order Runge-Kutta
			solver (using Heun's method).
			@param currentDelta		Current fiber direction.
			@param nextDelta		Next fiber direction.
			@param currentPosition	Current fiber position.
			@param nextPosition		Next fiber position.
			@param Csymbols			Christoffel symbols computed at "currentPosition. */

		void solverRK2Heun(double * currentDelta, double * nextDelta, double * currentPosition, double * nextPosition, double * Csymbols);

		/** Compute the next position and direction using a second-order Runge-Kutta
			solver (using the midpoint method).
			@param currentDelta		Current fiber direction.
			@param nextDelta		Next fiber direction.
			@param currentPosition	Current fiber position.
			@param nextPosition		Next fiber position.
			@param Csymbols			Christoffel symbols computed at "currentPosition. */

		void solverRK2Midpoint(double * currentDelta, double * nextDelta, double * currentPosition, double * nextPosition, double * Csymbols);

		/** Compute the next position and direction using a fourth-order Runge-Kutta solver.
			@param currentDelta		Current fiber direction.
			@param nextDelta		Next fiber direction.
			@param currentPosition	Current fiber position.
			@param nextPosition		Next fiber position.
			@param Csymbols			Christoffel symbols computed at "currentPosition. */

		void solverRK4(double * currentDelta, double * nextDelta, double * currentPosition, double * nextPosition, double * Csymbols);

		/** Compute the fiber direction ("delta") in the next fiber point. Adds the
			"factor" (which is equal to the step size) multiplied by a value computed
			from the "slope" vector and Christoffel symbols to the current direction.
			Output is written to the "o" vector. The slope vector is a fiber direction; 
			for simple Euler steps, it is equal to the actual fiber direction, but for
			ODE solvers of higher orders, it may be equal to an intermediate direction.
			@param current			Current fiber direction.
			@param slope			Fiber direction used to compute the second derivative. 
			@param Csymbols			Christoffel symbols.
			@param factor			Step size.
			@param o				Output direction. */

		void computeDelta(double * current, double * slope, double * Csymbols, double factor, double * o);

		/** Round a double to the nearest integer. 
			@param x				Input value. */

		int round(double x);

		/** Length of the diagonal of a cell. Used in "computeDelta" to avoid steps
			larger than one cell diagonal. */

		double cellDiagonal;

		/** ODE solver used to compute the fibers. */

		vtkFiberTrackingGeodesicFilter::ODESolver mySolver;

}; // class geodesicFiberTracker


} // namespace bmia


#endif // bmia_FiberTrackingPlugin_geodesicFiberTracker_h