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
