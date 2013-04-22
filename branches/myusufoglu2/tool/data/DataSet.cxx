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
 * DataSet.cxx
 *
 * 2009-10-15	Tim Peeters
 * - First version
 *
 * 2010-09-29	Evert van Aart
 * - Added "updateData" function.
 *
 * 2010-11-10	Tim Peeters
 * - Add setName() function.
 */

#include "DataSet.h"
#include "Attributes.h"

#include <vtkImageData.h>
#include <vtkPolyData.h>

#include <QString>

namespace bmia {
namespace data {

DataSet::DataSet(QString name, QString kind, vtkObject* obj)
{
    this->name = name;
    this->kind = kind;
    this->vtkData = obj;
    Q_ASSERT(this->vtkData);
    if (this->vtkData) this->vtkData->Register(NULL);
    this->attributes = new Attributes();
}

DataSet::~DataSet()
{
    if (this->vtkData) this->vtkData->UnRegister(NULL);
    this->vtkData = NULL;
    delete this->attributes;
}

void DataSet::updateData(vtkObject * obj)
{
	Q_ASSERT(obj);
	if (this->vtkData)
	{
		this->vtkData->UnRegister(NULL);
	}
	this->vtkData = obj;
	this->vtkData->Register(NULL);
}

QString DataSet::getName()
{
    return this->name;
}

void DataSet::setName(QString newname)
{
    this->name = newname;
}

QString DataSet::getKind()
{
    return this->kind;
}

Attributes* DataSet::getAttributes()
{
    return this->attributes;
}

vtkObject* DataSet::getVtkObject()
{
    return this->vtkData;
}

vtkDataSet* DataSet::getVtkDataSet()
{
    vtkObject* object = this->getVtkObject();
    if (!object) return NULL;
    vtkDataSet* ds = vtkDataSet::SafeDownCast(object);
    if (!ds) return NULL;
//    ds->Update();
    return ds;
}

vtkImageData* DataSet::getVtkImageData()
{
    vtkDataSet* ds = this->getVtkDataSet();
    if (!ds) return NULL;
    // this returns the down casted object, or
    // NULL if the object is not a vtkImageData.
    return vtkImageData::SafeDownCast(ds);
}

vtkPolyData* DataSet::getVtkPolyData()
{
    vtkDataSet* ds = this->getVtkDataSet();
    // this returns the down casted object, or
    // NULL if the object is not a vtkPolyData
    return vtkPolyData::SafeDownCast(ds);
}

} // namespace data
} // namespace bmia
