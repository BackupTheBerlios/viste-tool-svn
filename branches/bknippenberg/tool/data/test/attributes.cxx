/*
 * attributes.cxx
 *
 * 2009-10-30	Tim Peeters
 * - First version
 */

#include "data/DataSet.h"
#include "data/Attributes.h"

#include <QTextStream>
#include <QList>

using namespace bmia;
using namespace data;

/**
 * Test data attributes.
 */
int main(int argc, char ** argv)
{
    QTextStream out(stdout);

    DataSet* data = new DataSet("testData", "testKind");
    Attributes* attr = data->getAttributes();
    attr->addAttribute("number of legs per person", 2);
    attr->addAttribute("12+1",13);
    attr->addAttribute("b-value", 1000.0);
    int onetwothree = 123;
    attr->addAttribute("another integer", onetwothree);
    attr->addAttribute("and one more!", onetwothree);

    QList<double> v;
    v.push_back(12.0); v.push_back(15.2); v.push_back(2000.0);
    attr->addAttribute("some vector",v);

    double b; bool success;
    success = attr->getAttribute("b-value", b);
    out<<"success = "<<success<<", b = "<<b<<endl;
    success = attr->getAttribute("c-value", b);
    out<<"success = "<<success<<", c = "<<b<<endl;
    success = attr->getAttribute("b-value", b);
    out<<"success = "<<success<<", b = "<<b<<endl;

    v.push_back(123.456); v.push_back(987.654);
    attr->addAttribute("some vector",v);
    QList<double> w;
    success = attr->getAttribute("some vector", w);
    out<<"success = "<<success<<", vector size = "<<w.size()<<endl;

    attr->printAllAttributes();

    return (b != 1000.0); // pass the test if b == 1000.0
}
