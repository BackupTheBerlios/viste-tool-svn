#ifndef __DTIUtils_h
#define __DTIUtils_h

#include <iostream>
#include <fstream>

#include <string.h>
#include <stdlib.h>
#include <math.h>

using namespace std;

// Global variables for error/debug logging.
#define ALWAYS 0
#define ERROR  1
#define DEBUG  2

//------------------------------------------------------------------------------
// Set/Get macro.
//------------------------------------------------------------------------------
#define __DTIMACRO_SETGET(Value, Type) \
	virtual void Set##Value(Type _Arg) \
	{\
		if(this->Value != _Arg) \
			this->Value = _Arg;\
	}\
	virtual Type Get##Value()\
	{\
		return this->Value;\
	}\

//------------------------------------------------------------------------------
// Set/Is macro for boolean values.
//------------------------------------------------------------------------------
#define __DTIMACRO_SETIS(Value, Type) \
	virtual void Set##Value(Type _Arg) \
	{\
		if(this->Value != _Arg) \
			this->Value = _Arg;\
	}\
	virtual Type Is##Value()\
	{\
		return this->Value;\
	}\

//------------------------------------------------------------------------------
// Log macro. Logs only if Type <= LogLevel.
//------------------------------------------------------------------------------
#define __DTIMACRO_LOG(Msg, Type, LogLevel)\
	if(Type <= LogLevel)\
		cout << Msg;\

//---------------------------------------------------------------------------
//! \file   DTIUtils.h
//! \class  DTIUtils
//! \author Ralph Brecheisen
//! \brief  Contains number of static utility functions.
//---------------------------------------------------------------------------
class DTIUtils
{
public:

	//! Static log level variable that can be accessed by all.
	static int LogLevel;
	static const int USHORT = 1;
	static const int FLOAT = 2;

	//-------------------------------------------------------------------
	//! Converts integer to string.
	//-------------------------------------------------------------------
	static char *IntegerToString(int value)
	{
		char * str = new char[256];
		sprintf( str, "%d", value );
		return str;
		//return _itoa(value, str, 10);
	}

	//-------------------------------------------------------------------
	//! Converts long (64-bit integer) to string.
	//-------------------------------------------------------------------
	static char *LongToString(long value)
	{
		char * str = new char[256];
		sprintf( str, "%ld", value );
		return str;
		//return _ltoa(value, str, 10);
	}

	//-------------------------------------------------------------------
	//! Converts double to string with given number of digits.
	//-------------------------------------------------------------------
	static char *DoubleToString(double value, int numberDigits)
	{
		char * str = new char[256];
		sprintf( str, "%lf", value );
		return str;
		//return _ecvt(value, numberDigits, 0, 0);
	}

	/** Converts array of unsigned characters to null-terminated string */
	static char *UnsignedCharArrayToString(unsigned char *array, unsigned long len)
	{
			char *str = new char[len+1];
			memcpy(str, array, len*sizeof(unsigned char));
			str[len] = '\0';
			return str;
	}

	/** Converts string to integer */
	static int StringToInteger(char *str)
	{
		if(str == 0)
			return 0;
		return atoi(str);
	}

	/** Converts string to long */
	static long StringToLong(char *str)
	{
		if(str == 0)
			return 0;
		return atol(str);
	}

	/** Converts string to double */
	static double StringToDouble(char *str)
	{
		if(str == 0)
			return 0;
		return atof(str);
	}

	/** Converts string to bool */
	static bool StringToBool(char *str)
	{
		if(str == 0)
			return false;
		str = DTIUtils::TrimR(str);
		if(strcmp(str, "true") == 0 || strcmp(str, "True") == 0 || strcmp(str, "TRUE") == 0 || strcmp(str, "1") == 0)
			return true;
		if(strcmp(str, "false") == 0 || strcmp(str, "False") == 0 || strcmp(str, "FALSE") == 0 || strcmp(str, "0") == 0)
			return false;
		return false;
	}

	/** Converts delimiter-separated list of strings to integer array */
	static int *StringToIntegerArray(char *str, char *delim = ",")
	{
		if(str == 0)
			return 0;
		str = DTIUtils::TrimR(str);
		int *list = new int[128];
		char *token = strtok(str, delim);
		int i = 0;
		while(token != 0)
		{
			list[i] = DTIUtils::StringToInteger(DTIUtils::Trim1(token));
			token = strtok(0, delim);
			i++;
		}
		int *out = new int[i];
		memcpy(out, list, i*sizeof(int));
		delete [] list;
		return out;
	}

	/** Converts delimiter-separated list of strings to double array */
	static double *StringToDoubleArray(char *str, char *delim = ",")
	{
		if(str == 0)
			return 0;
		str = DTIUtils::TrimR(str);
		double *list = new double[128];
		char *token = strtok(str, delim);
		int i = 0;
		while(token != 0)
		{
			list[i] = DTIUtils::StringToDouble(DTIUtils::Trim1(token));
			token = strtok(0, delim);
			i++;
		}
		double *out = new double[i];
		memcpy(out, list, i*sizeof(double));
		delete [] list;
		return out;
	}

	/** Converts delimiter-separated list of strings to double array */
	static char **StringToStringArray(char *str, char *delim = ",")
	{
		if(str == 0)
			return 0;
		str = DTIUtils::TrimR(str);

		char **list = new char*[128];
		char *token = strtok(str, delim);
		int i = 0;
		while(token != 0)
		{
			list[i] = DTIUtils::Trim1(token);
			token = strtok(0, delim);
			i++;
		}
		char **out = new char*[i];
		for(int j = 0; j < i; j++)
		{
			int len = strlen(list[j]);
			out[j] = new char[len+1];

			memcpy(out[j], list[j], len*sizeof(char));
			out[j][len] = '\0';
		}

		return out;
	}

	/** Counts number of digits for the given integer */
	static int GetNumberDigits(int value)
	{
		return (int) strlen(DTIUtils::IntegerToString(value));
	}

	/** Counts number of digits for the given long */
	static int GetNumberDigits(long value)
	{
		return (int) strlen(DTIUtils::IntegerToString(value));
	}

	/** Build indexed filename */
	static char *BuildIndexedFileName(char *path, char *prefix, int index, int nrdigits, char *postfix)
	{
		path = DTIUtils::TrimR(path);
		prefix = DTIUtils::TrimR(prefix);
		postfix = DTIUtils::TrimR(postfix);

		// Check if an extension was provided
		int pflen = 0;
		if(postfix != 0 && strlen(postfix) > 0)
			if(postfix[0] == '.')
				pflen = strlen(postfix);
			else
				pflen = strlen(postfix) + 1;

		char *fileName = new char[strlen(path) + strlen(prefix) + nrdigits + pflen + 1];
		strcpy(fileName, path);

		// If path end with slashes, append prefix right away, otherwise insert slashes
		int pathlen = strlen(path);
		if(path[pathlen-1] == '\\' || path[pathlen-1] == '/')
			strcat(fileName, prefix);
		else
		{
			strcat(fileName, "\\");
			strcat(fileName, prefix);
		}

		// Pad with zeros if necessary
                int nrIndexDigits = DTIUtils::GetNumberDigits(index);
                if(nrdigits > 0)
                {
		        for(int i = 0; i < (nrdigits - nrIndexDigits); i++)
			        strcat(fileName, "0");
                }

		strcat(fileName, DTIUtils::IntegerToString(index));

		// See if an extension needs to be added to the filename
		if(postfix != 0 && strlen(postfix) > 0)
		{
			if(postfix[0] != '.')
				strcat(fileName, ".");

			strcat(fileName, postfix);
		}

		return fileName;
	}

	/** Checks for existence of file */
	static bool FileExists(char *fileName)
	{
		fileName = DTIUtils::TrimR(fileName);
		bool result = true;

		ifstream istr;
		istr.open(fileName, ios::in);

		if(!istr)
			result = false;

		istr.close();
		return result;
	}

        /** Calculate minimum value of dataset (double) */
        static unsigned short GetMinValue(unsigned short *data, int len)
        {
                unsigned short minvalue = (unsigned short) 99999999;
                for(int i = 0; i < len; i++)
                        if(data[i] < minvalue)
                                minvalue = data[i];
                return minvalue;
        }

        /** Calculate maximum value of dataset (double) */
        static unsigned short GetMaxValue(unsigned short *data, int len)
        {
                unsigned short maxvalue = 0;
                for(int i = 0; i < len; i++)
                        if(data[i] > maxvalue)
                                maxvalue = data[i];
                return maxvalue;
        }

	/** Trim leading and trailing spaces from string */
	static char *Trim(char *str)
	{
		if(str == 0)
			return str;
		str = DTIUtils::TrimR(str);

		int len = strlen(str);
		if(len == 0)
			return str;

		// If string has no leading and trailing spaces, return it as is
		if(str[0] != ' ' && str[len-1] != ' ')
			return str;

		int begin = 0;
		int end   = len-1;

		// Find first and last non-empty indices.
		while(str[begin] == ' ')
			begin++;
		while(str[end] == ' ')
			end--;

		int newlen = end-begin+2;
		char *trimmed = new char[newlen];
		strncpy(trimmed, str+begin, end-begin+1);
		trimmed[newlen] = '\0';

		return trimmed;
	}

	static char *Trim1(char *str)
	{
		str = DTIUtils::TrimR(str);
		int len = strlen(str);
		int begin = 0;
		int end = len-1;

		if(str[len-1] == '\n')
			end = len-2;

		if(str[begin] != ' ' && str[end] != ' ')
			return str;
		while(str[begin] == ' ') begin++;
		while(str[end] == ' ') end--;
		int newlen = end-begin+1;
		char *trimmed = new char[newlen+1];
		memcpy(trimmed, str+begin, newlen);
		trimmed[newlen] = '\0';
		return trimmed;
	}

	/** Appends back/forward slash to directory path (if needed) */
	static char *AppendSlashToPath(char *path)
	{
		path = DTIUtils::TrimR(path);
		int len = strlen(path);

		if(path[len-1] != '\\' && path[len-1] != '/')
		{
			// Find out whether back or forward slashes were used.
			bool back = true;

			for(int i = 0; i < len; i++)
			{
				if(path[i] == '/')
				{
					back = false;
					break;
				}
			}

			char *tmp = new char[len+1];
			strcpy(tmp, path);

			if(back)
				tmp[len] = '\\';
			else
				tmp[len] = '/';

			tmp[len+1] = '\0';

			// Do not just delete this array. You don't know
			// where it came from. For example, if it comes from
			// a std::string::c_str() operation, then you cannot
			// just delete it!

			//delete [] path;

			return tmp;
		}

		return path;
	}

	/** Inserts string A at given position in string B */
	static char *InsertCharacter(char *stra, char *strb, int pos)
	{
		if(stra == 0)
			return strb;
		if(strb == 0)
			return stra;
		stra = DTIUtils::TrimR(stra);
		strb = DTIUtils::TrimR(strb);

		int lena = strlen(stra);
		int lenb = strlen(strb);

		if(pos > lenb-1)
			return strb;

		char *tmp = new char[lena+lenb+1];
	}

	/** Concatenate two strings */
	static char *Concatenate(char *stra, char *strb)
	{
		stra = DTIUtils::TrimR(stra);
		strb = DTIUtils::TrimR(strb);

		int lena  = strlen(stra);
		int lenb  = strlen(strb);
		char *tmp = new char[lena+lenb+1];

		strcpy(tmp, stra);
		strcpy(tmp+lena, strb);

		tmp[lena+lenb] = '\0';
		return tmp;
	}

	/** Write pixels to raw data file */
	static void WriteToFile(char *filename, int rows, int columns, unsigned short *pixels)
	{
		filename = DTIUtils::TrimR(filename);
		FILE *f = fopen(filename, "wb");
		fwrite(pixels, sizeof(unsigned short), rows*columns, f);
		fclose(f);
	}

	/** Removes windows \r character */
	static char * TrimR( char * str )
	{
		if( str == 0 ) return str;
		int i = strlen(str);
		if( str[i-1] == '\r' )
			str[i-1] = '\0';
		return str;
	}

	/** Convert string to lower-case */
	static char * ToLower( char * str )
	{
		char * p = DTIUtils::TrimR(str);
		while( (*p) = tolower( (*p) ) )
			p++;
		return str;
	}
};

#endif
