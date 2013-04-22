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

#include "vtkShaderBase.h"
#include <vtkgl.h>
#include <iostream>
#include <string>

///////////////////////////////////////////////////////////////////
vtkShaderBase::vtkShaderBase() : m_iProgramId( 0 ), m_pAttribs( 0 )
{
}

///////////////////////////////////////////////////////////////////
vtkShaderBase::~vtkShaderBase()
{
	if( m_iProgramId > 0 )
                vtkgl::DeleteProgram( m_iProgramId );
}

///////////////////////////////////////////////////////////////////
void vtkShaderBase::Activate()
{
    vtkgl::UseProgram( m_iProgramId );
}

///////////////////////////////////////////////////////////////////
void vtkShaderBase::Deactivate()
{
    vtkgl::UseProgram( 0 );
}

///////////////////////////////////////////////////////////////////
void vtkShaderBase::Load()
{
	if( m_iProgramId > 0 )
            vtkgl::DeleteProgram( m_iProgramId );

	std::string str;

	str = GetVertexShader();
	const char * vText = str.c_str();

        unsigned int vid = vtkgl::CreateShader( vtkgl::VERTEX_SHADER );

        vtkgl::ShaderSource( vid, 1, & vText, 0 );
        vtkgl::CompileShader( vid );
	CheckErrors( vid );


	str = GetFragShader();
	const char * fText = str.c_str();

        unsigned int fid = vtkgl::CreateShader( vtkgl::FRAGMENT_SHADER );
        vtkgl::ShaderSource( fid, 1, & fText, 0 );
        vtkgl::CompileShader( fid );
	CheckErrors( fid );

        m_iProgramId = vtkgl::CreateProgram();

	if( m_pAttribs && ! m_pAttribs->empty() )
	{
		int idx = 1;
		std::vector<std::string>::iterator iter = m_pAttribs->begin();
		for( ; iter != m_pAttribs->end(); iter++ )
                        vtkgl::BindAttribLocation( m_iProgramId, idx++, iter->c_str() );
	}

        vtkgl::AttachShader( m_iProgramId, vid );
        vtkgl::AttachShader( m_iProgramId, fid );
        vtkgl::LinkProgram( m_iProgramId );

	Deactivate();
}

///////////////////////////////////////////////////////////////////
void vtkShaderBase::LoadFromFile( const char * vFile, const char * fFile )
{
	std::ifstream file;
	file.open( vFile, ios::in );
	if( file.fail() )
	{
		std::cout << "Could not open file " << vFile << std::endl;
		return;
	}

	int length = 0;
	file.seekg( 0, ios::end );
	length = file.tellg();
	file.seekg( 0, ios::beg );
	
	char * vText = new char[length];
	file.read( vText, length );
	file.close();

	file.open( fFile, ios::in );
	if( file.fail() )
	{
		std::cout << "Could not open file " << fFile << std::endl;
		return;
	}

	file.seekg( 0, ios::end );
	length = file.tellg();
	file.seekg( 0, ios::beg );

	char * fText = new char[length];
	file.read( fText, length );
	file.close();

	LoadFromText( (const char *) vText, (const char *) fText );
}

///////////////////////////////////////////////////////////////////
void vtkShaderBase::LoadFromText( const char * vText, const char * fText )
{
	if( m_iProgramId > 0 )
                vtkgl::DeleteProgram( m_iProgramId );

        unsigned int vid = vtkgl::CreateShader( vtkgl::VERTEX_SHADER );
        vtkgl::ShaderSource( vid, 1, & vText, 0 );
        vtkgl::CompileShader( vid );
	CheckErrors( vid );

        unsigned int fid = vtkgl::CreateShader( vtkgl::FRAGMENT_SHADER );
        vtkgl::ShaderSource( fid, 1, & fText, 0 );
        vtkgl::CompileShader( fid );
	CheckErrors( fid );

        m_iProgramId = vtkgl::CreateProgram();

        vtkgl::AttachShader( m_iProgramId, vid );
        vtkgl::AttachShader( m_iProgramId, fid );
        vtkgl::LinkProgram( m_iProgramId );
	CheckErrors( m_iProgramId );

	Deactivate();
}

///////////////////////////////////////////////////////////////////
unsigned int vtkShaderBase::GetId()
{
	return m_iProgramId;
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::CheckErrors( unsigned int id )
{
	char shaderLog[1024];

        vtkgl::GetShaderInfoLog( id, 1024, 0, shaderLog );
	if( * shaderLog )
		std::cout << shaderLog << std::endl;
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::SetFloat1( float f1, const char * name )
{
        vtkgl::Uniform1f(
                vtkgl::GetUniformLocation( m_iProgramId, name ), f1 );
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::SetFloat2( float f1, float f2, const char * name )
{
        vtkgl::Uniform2f(
                vtkgl::GetUniformLocation( m_iProgramId, name ), f1, f2 );
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::SetFloat3( float f1, float f2, float f3, const char * name )
{
        vtkgl::Uniform3f(
                vtkgl::GetUniformLocation( m_iProgramId, name ), f1, f2, f3 );
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::SetFloat4( float f1, float f2, float f3, float f4, const char * name )
{
        vtkgl::Uniform4f(
                vtkgl::GetUniformLocation( m_iProgramId, name ), f1, f2, f3, f4 );
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::SetFloatArray1( float * fArray, int count, const char * name )
{
        vtkgl::Uniform1fv(
                vtkgl::GetUniformLocation( m_iProgramId, name ), count, fArray );
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::SetFloatArray2( float * fArray, int count, const char * name )
{
        vtkgl::Uniform2fv(
                vtkgl::GetUniformLocation( m_iProgramId, name ), count, fArray );
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::SetFloatArray3( float * fArray, int count, const char * name )
{
        vtkgl::Uniform3fv(
                vtkgl::GetUniformLocation( m_iProgramId, name ), count, fArray );
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::SetFloatArray4( float * fArray, int count, const char * name )
{
        vtkgl::Uniform4fv(
                vtkgl::GetUniformLocation( m_iProgramId, name ), count, fArray );
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::SetInt1( int i1, const char * name )
{
        vtkgl::Uniform1i(
                vtkgl::GetUniformLocation( m_iProgramId, name ), i1 );
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::SetInt2( int i1, int i2, const char * name )
{
        vtkgl::Uniform2i(
                vtkgl::GetUniformLocation( m_iProgramId, name ), i1, i2 );
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::SetInt3( int i1, int i2, int i3, const char * name )
{
        vtkgl::Uniform3i(
                vtkgl::GetUniformLocation( m_iProgramId, name ), i1, i2, i3 );
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::SetInt4( int i1, int i2, int i3, int i4, const char * name )
{
        vtkgl::Uniform4i(
                vtkgl::GetUniformLocation( m_iProgramId, name ), i1, i2, i3, i4 );
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::SetIntArray1( int * iArray, int count, const char * name )
{
        vtkgl::Uniform1iv(
                vtkgl::GetUniformLocation( m_iProgramId, name ), count, iArray );
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::SetIntArray2( int * iArray, int count, const char * name )
{
        vtkgl::Uniform2iv(
                vtkgl::GetUniformLocation( m_iProgramId, name ), count, iArray );
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::SetIntArray3( int * iArray, int count, const char * name )
{
        vtkgl::Uniform3iv(
                vtkgl::GetUniformLocation( m_iProgramId, name ), count, iArray );
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::SetIntArray4( int * iArray, int count, const char * name )
{
        vtkgl::Uniform4iv(
                vtkgl::GetUniformLocation( m_iProgramId, name ), count, iArray );
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::SetBool1( bool b1, const char * name )
{
	int value = b1 ? 1 : 0;
        vtkgl::Uniform1i(
                vtkgl::GetUniformLocation( m_iProgramId, name ), value );
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::SetBool2( bool b1, bool b2, const char * name )
{
	int i1 = b1 ? 1 : 0;
	int i2 = b2 ? 1 : 0;
        vtkgl::Uniform2i(
                vtkgl::GetUniformLocation( m_iProgramId, name ), i1, i2 );
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::SetBool3( bool b1, bool b2, bool b3, const char * name )
{
	int i1 = b1 ? 1 : 0;
	int i2 = b2 ? 1 : 0;
	int i3 = b3 ? 1 : 0;
        vtkgl::Uniform3i(
                vtkgl::GetUniformLocation( m_iProgramId, name ), i1, i2, i3 );
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::SetBool4( bool b1, bool b2, bool b3, bool b4, const char * name )
{
	int i1 = b1 ? 1 : 0;
	int i2 = b2 ? 1 : 0;
	int i3 = b3 ? 1 : 0;
	int i4 = b4 ? 1 : 0;
        vtkgl::Uniform4i(
                vtkgl::GetUniformLocation( m_iProgramId, name ), i1, i2, i3, i4 );
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::SetBoolArray1( bool * bArray, int count, const char * name )
{
	int * iArray = new int[count];
	for( int i = 0; i < count; i++ )
		iArray[i] = bArray[i] ? 1 : 0;
        vtkgl::Uniform1iv(
                vtkgl::GetUniformLocation( m_iProgramId, name ), count, iArray );
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::SetBoolArray2( bool * bArray, int count, const char * name )
{
	int * iArray = new int[count];
	for( int i = 0; i < count; i++ )
		iArray[i] = bArray[i] ? 1 : 0;
        vtkgl::Uniform2iv(
                vtkgl::GetUniformLocation( m_iProgramId, name ), count, iArray );
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::SetBoolArray3( bool * bArray, int count, const char * name )
{
	int * iArray = new int[count];
	for( int i = 0; i < count; i++ )
		iArray[i] = bArray[i] ? 1 : 0;
        vtkgl::Uniform3iv(
                vtkgl::GetUniformLocation( m_iProgramId, name ), count, iArray );
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::SetBoolArray4( bool * bArray, int count, const char * name )
{
	int * iArray = new int[count];
	for( int i = 0; i < count; i++ )
		iArray[i] = bArray[i] ? 1 : 0;
        vtkgl::Uniform4iv(
                vtkgl::GetUniformLocation( m_iProgramId, name ), count, iArray );
}

/////////////////////////////////////////////////////////////////////////////////////////
void vtkShaderBase::AddAttrib( std::string strName )
{
	if( ! m_pAttribs )
		m_pAttribs = new std::vector<std::string>;
	m_pAttribs->push_back( strName );
}

/////////////////////////////////////////////////////////////////////////////////////////
std::string vtkShaderBase::GetAttrib( unsigned int iIdx )
{
	if( ! m_pAttribs )
		return std::string( "" );
	if( iIdx > m_pAttribs->size() - 1 )
		return std::string( "" );
	return m_pAttribs->at( iIdx );
}
