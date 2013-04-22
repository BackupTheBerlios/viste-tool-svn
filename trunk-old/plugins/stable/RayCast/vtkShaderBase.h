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

#ifndef __vtkShaderBase_h
#define __vtkShaderBase_h

#include <vtkObject.h>
#include <string>
#include <vector>

class vtkShaderBase : public vtkObject
{
public:

	void Activate();
	void Deactivate();

	void Load();
	void LoadFromFile( const char * vFile, const char * fFile );
	void LoadFromText( const char * vText, const char * fText );

	void AddAttrib( std::string strName );
	std::string GetAttrib( unsigned int iIdx );

	void SetFloat1( float f1, const char * name );
	void SetFloat2( float f1, float f2, const char * name );
	void SetFloat3( float f1, float f2, float f3, const char * name );
	void SetFloat4( float f1, float f2, float f3, float f4, const char * name );

	void SetFloatArray1( float * fArray, int count, const char * name );
	void SetFloatArray2( float * fArray, int count, const char * name );
	void SetFloatArray3( float * fArray, int count, const char * name );
	void SetFloatArray4( float * fArray, int count, const char * name );

	void SetInt1( int i1, const char * name );
	void SetInt2( int i1, int i2, const char * name );
	void SetInt3( int i1, int i2, int i3, const char * name );
	void SetInt4( int i1, int i2, int i3, int i4, const char * name );
	
	void SetIntArray1( int * iArray, int count, const char * name );
	void SetIntArray2( int * iArray, int count, const char * name );
	void SetIntArray3( int * iArray, int count, const char * name );
	void SetIntArray4( int * iArray, int count, const char * name );

	void SetBool1( bool b1, const char * name );
	void SetBool2( bool b1, bool b2, const char * name );
	void SetBool3( bool b1, bool b2, bool b3, const char * name );
	void SetBool4( bool b1, bool b2, bool b3, bool b4, const char * name );

	void SetBoolArray1( bool * bArray, int count, const char * name );
	void SetBoolArray2( bool * bArray, int count, const char * name );
	void SetBoolArray3( bool * bArray, int count, const char * name );
	void SetBoolArray4( bool * bArray, int count, const char * name );

	unsigned int GetId();

protected:

	vtkShaderBase();
	virtual ~vtkShaderBase();
	virtual std::string GetVertexShader() = 0;
	virtual std::string GetFragShader() = 0;

	void CheckErrors( unsigned int id );

	std::vector<std::string> * m_pAttribs;

	unsigned int m_iProgramId;
};

#endif
