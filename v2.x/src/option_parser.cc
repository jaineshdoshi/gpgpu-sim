/*
 * option_parser.cc
 *
 * Copyright © 2009 by Wilson W. L. Fung and the University of British Columbia, 
 * Vancouver, BC V6T 1Z4, All Rights Reserved.
 *
 * THIS IS A LEGAL DOCUMENT BY DOWNLOADING GPGPU-SIM, YOU ARE AGREEING TO THESE
 * TERMS AND CONDITIONS.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * NOTE: The files libcuda/cuda_runtime_api.c and src/cuda-sim/cuda-math.h
 * are derived from the CUDA Toolset available from http://www.nvidia.com/cuda
 * (property of NVIDIA).  The files benchmarks/BlackScholes/ and
 * benchmarks/template/ are derived from the CUDA SDK available from
 * http://www.nvidia.com/cuda (also property of NVIDIA).  The files from
 * src/intersim/ are derived from Booksim (a simulator provided with the
 * textbook "Principles and Practices of Interconnection Networks" available
 * from http://cva.stanford.edu/books/ppin/). As such, those files are bound by
 * the corresponding legal terms and conditions set forth separately (original
 * copyright notices are left in files from these sources and where we have
 * modified a file our copyright notice appears before the original copyright
 * notice).
 *
 * Using this version of GPGPU-Sim requires a complete installation of CUDA
 * which is distributed seperately by NVIDIA under separate terms and
 * conditions.  To use this version of GPGPU-Sim with OpenCL requires a
 * recent version of NVIDIA's drivers which support OpenCL.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the University of British Columbia nor the names of
 * its contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * 4. This version of GPGPU-SIM is distributed freely for non-commercial use only.
 *
 * 5. No nonprofit user may place any restrictions on the use of this software,
 * including as modified by the user, by any other authorized user.
 *
 * 6. GPGPU-SIM was developed primarily by Tor M. Aamodt, Wilson W. L. Fung,
 * Ali Bakhoda, George L. Yuan, at the University of British Columbia,
 * Vancouver, BC V6T 1Z4
 */


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <list>
#include <map>
#include <string.h>

using namespace std;

// A generic option registry regardless of data type
class OptionRegistryInterface 
{
public:
   OptionRegistryInterface(const string optionName, const string optionDesc) 
      : m_optionName(optionName), m_optionDesc(optionDesc) 
   {}

   virtual ~OptionRegistryInterface() {}

   const string& GetName() { return m_optionName; }
   const string& GetDesc() { return m_optionDesc; }
   virtual string toString() = 0;
   virtual bool fromString(const string str) = 0;
   virtual bool isFlag() = 0;
    virtual bool assignDefault(const char *str) = 0;

private:
   string m_optionName;
   string m_optionDesc;
};

// Template for option registry - class T = specify data type of the option
template <class T>
class OptionRegistry : public OptionRegistryInterface 
{
public:
   OptionRegistry(const string name, const string desc, T &variable)
      : OptionRegistryInterface(name, desc), m_variable(variable)
   {}

   virtual ~OptionRegistry() {}

   virtual string toString() 
   {
      stringstream ss;
      ss << m_variable;
      return ss.str();
   }

   virtual bool fromString(const string str)
   {
      stringstream ss(str);
      ss.exceptions(stringstream::failbit | stringstream::badbit);
      ss << setbase(10);
      if (str.size() > 1 && str[0] == '0') {
         if (str.size() > 2 && str[1] == 'x') { 
            ss.ignore(2);
            ss << setbase(16);
         } else {
            ss.ignore(1);
            ss << setbase(8);
         }
      }
      try {
         ss >> m_variable;
      } catch (stringstream::failure &e) {
         return false;
      }
      return true;
   }

   virtual bool isFlag() { return false; }
    virtual bool assignDefault(const char *str) { return fromString(str); }

   operator T()
   {
      return m_variable;
   }

private:
   T &m_variable;
};

// specialized parser for string-type options
template<>
bool OptionRegistry<string>::fromString(const string str)
{
   m_variable = str;
   return true;
}

// specialized parser for c-string type options
template<>
bool OptionRegistry<char *>::fromString(const string str)
{
   m_variable = new char[str.size() + 1];
   strcpy(m_variable, str.c_str());
   return true;
}

// specialized default assignment for c-string type option to allow NULL default
template<>
bool OptionRegistry<char *>::assignDefault(const char *str) 
{
    m_variable = const_cast<char *>(str); // c-string options are not meant to be edited anyway
    return true;
}

// specialized default assignment for c-string type option to allow NULL default
template<>
string OptionRegistry<char *>::toString() 
{
    stringstream ss;
    if (m_variable != NULL) {
        ss << m_variable;
    } else {
        ss << "NULL";
    }
    return ss.str();
}

// specialized parser for boolean options 
template<>
bool OptionRegistry<bool>::fromString(const string str)
{
   int value = 1;
   bool parsed = true;
   stringstream ss(str);
   ss.exceptions(stringstream::failbit | stringstream::badbit);
   try {
      ss >> value;
   } catch (stringstream::failure &ep) {
      parsed = false;
   }
   assert(value == 0 or value == 1); // sanity check for boolean options (it can only be 1 or 0)
   m_variable = (value != 0);
   return parsed;
}

// specializing a flag query function to identify boolean option
template<>
bool OptionRegistry<bool>::isFlag() { return true; }

// class holding a collection of options and parse them from command line/configfile
class OptionParser
{
public:
   OptionParser() {}
   ~OptionParser() 
   {
      OptionCollection::iterator i_option;
      for (i_option = m_optionReg.begin(); i_option != m_optionReg.end(); ++i_option) {
         delete (*i_option);
      }
   } 

   template<class T>
   void Register(const string optionName, const string optionDesc, T &optionVariable, const char *optionDefault)
   {
      OptionRegistry<T> *p_option = new OptionRegistry<T>(optionName, optionDesc, optionVariable);
      m_optionReg.push_back(p_option);
      m_optionMap[optionName] = p_option;
        p_option->assignDefault(optionDefault);
   }

   void ParseCommandLine(int argc, const char * const argv[]) 
   {
      for (int i = 1; i < argc; i++) {
         OptionMap::iterator i_option;
         bool optionFound = false;

         i_option = m_optionMap.find(argv[i]);
         if (i_option != m_optionMap.end()) {
            const char *argstr = (i + 1 < argc)? argv[i + 1] : "";
            OptionRegistryInterface *p_option = i_option->second;
            if (p_option->isFlag()) { 
               if (p_option->fromString(argstr) == true) {
                  i += 1;
               } 
            } else { 
               if (p_option->fromString(argstr) == false) {
                  fprintf(stderr, "\n\nGPGPU-Sim ** ERROR: Cannot parse value '%s' for option '%s'.\n", argstr, argv[i]);
                  exit(1);
               }
               i += 1;
            }
            optionFound = true;
         } else if (string(argv[i]) == "-config") {
            if (i + 1 >= argc) {
               fprintf(stderr, "\n\nGPGPU-Sim ** ERROR: Missing filename for option '-config'.\n");
               exit(1);
            }
            ParseFile(argv[i + 1]);
            i += 1;
            optionFound = true;
         }

         if (optionFound == false) {
            fprintf(stderr, "\n\nGPGPU-Sim ** ERROR: Unknown Option: '%s' \n", argv[i]);
            exit(1);
         }
      }
   }

   void ParseFile(const char *filename) {
      ifstream inputFile;
      stringstream args;

      // open config file, stream every line into a continuous buffer 
      // get rid of comments in the process
      inputFile.open(filename);
      if (!inputFile.good()) {
         fprintf(stderr, "\n\nGPGPU-Sim ** ERROR: Cannot open config file '%s'\n", filename);
         exit(1);
      }
      while (inputFile.good()) {
         string line;
         getline(inputFile, line);
         size_t commentStart = line.find_first_of("#");
         if (commentStart != line.npos) {
            line.erase(commentStart);
         }
         args << line << ' ';
      }
      inputFile.close();

      // extract non-whitespace string tokens
      vector<char*> argv;
      argv.push_back(new char[6]); 
      strcpy(argv[0], "dummy");
      while (args.good()) {
         string argNew;
         args >> argNew;

         if (argNew.size() == 0) continue; // this is probably the last token

         if (argNew[0] == '"') {
            while (args.good() && argNew[argNew.size()-1] != '"') {
               string argCont;
               args >> argCont;
               argNew += " " + argCont;
            }
            argNew.erase(0,1);
            argNew.erase(argNew.size()-1);
         }

         char *c_argNew = new char[argNew.size() + 1];
         strcpy(c_argNew, argNew.c_str());
         argv.push_back(c_argNew);
      }

      // pass the string token into normal commandline parser
      char **targv = (char**)calloc(argv.size(), sizeof(char*));
      for( unsigned k=0; k < argv.size(); k++ ) 
         targv[k] = argv[k];
      ParseCommandLine(argv.size(), targv);
      free(targv);
      for (size_t i = 0; i < argv.size(); i++) {
         delete[] argv[i];
      }
   }

   void Print(FILE *fout)
   {
      OptionCollection::iterator i_option;
      for (i_option = m_optionReg.begin(); i_option != m_optionReg.end(); ++i_option) {
         stringstream sout;
         sout << setw(20) << left << (*i_option)->GetName() << " "; 
         sout << setw(20) << right << (*i_option)->toString() << " # ";
         sout << left << (*i_option)->GetDesc();
         sout << std::endl;
         fprintf(fout, "%s", sout.str().c_str());
      }
   }

private:
   typedef list<OptionRegistryInterface*> OptionCollection;
   OptionCollection m_optionReg;
   typedef map<string, OptionRegistryInterface*> OptionMap;
   OptionMap m_optionMap;
};

#include "option_parser.h"

option_parser_t option_parser_create() 
{
   OptionParser *p_opr = new OptionParser();
   return reinterpret_cast<option_parser_t>(p_opr);
}

void option_parser_destroy(option_parser_t opp)
{
   OptionParser *p_opr = reinterpret_cast<OptionParser *>(opp);
   delete p_opr;
}

void option_parser_register(option_parser_t opp, 
                            const char *name, 
                            enum option_dtype type, 
                            void *variable, 
                            const char *desc,  
                            const char *defaultvalue)
{
   OptionParser *p_opr = reinterpret_cast<OptionParser *>(opp);
   switch (type) {
      case OPT_INT32:     p_opr->Register<int>(name, desc, *(int*)variable, defaultvalue); break;
      case OPT_UINT32:    p_opr->Register<unsigned int>(name, desc, *(unsigned int*)variable, defaultvalue); break;
      case OPT_INT64:     p_opr->Register<long long>(name, desc, *(long long*)variable, defaultvalue); break;
      case OPT_UINT64:    p_opr->Register<unsigned long long>(name, desc, *(unsigned long long*)variable, defaultvalue); break;
      case OPT_BOOL:      p_opr->Register<bool>(name, desc, *(bool*)variable, defaultvalue); break;
      case OPT_FLOAT:     p_opr->Register<float>(name, desc, *(float*)variable, defaultvalue); break;
      case OPT_DOUBLE:    p_opr->Register<double>(name, desc, *(double*)variable, defaultvalue); break;
      case OPT_CSTR:      p_opr->Register<char*>(name, desc, *(char**)variable, defaultvalue); break;
        default:
            fprintf(stderr, "\n\nGPGPU-Sim ** ERROR: option data type (%d) not supported!\n", type);
            exit(1);
            break;
    }
}

void option_parser_cmdline(option_parser_t opp,
                           int argc, const char *argv[]) 
{
    OptionParser *p_opr = reinterpret_cast<OptionParser *>(opp);
    p_opr->ParseCommandLine(argc,argv);
}

void option_parser_cfgfile(option_parser_t opp,
                           const char *filename)
{
    OptionParser *p_opr = reinterpret_cast<OptionParser *>(opp);
    p_opr->ParseFile(filename);
}

void option_parser_print(option_parser_t opp, 
                         FILE *fout)
{
    OptionParser *p_opr = reinterpret_cast<OptionParser *>(opp);
    p_opr->Print(fout);
}



// #define UNIT_TEST
#ifdef UNIT_TEST

class testtype
{
public:
   int idata;
   float fdata;
   string sdata;
   unsigned long long ulldata;
   bool bdata;
   unsigned int boolint;
   char * coption;

   testtype() 
      : idata(0), 
        fdata(0.0f), 
        sdata(""),
        ulldata(0),
        bdata(false)
   { }
};


int cppinterfacetest(int argc, char *argv[])
{
   testtype c;
   OptionParser optionparser;
   c.idata = 123;
   c.fdata = 3249586.333;
   c.sdata = string("haha");

   optionparser.Register<int>("-idata", "integer data", c.idata, "-456");
   optionparser.Register<float>("-fdata", "floating point data", c.fdata, "0.001");
   optionparser.Register<string>("-sdata", "first string data", c.sdata, "hellow");
   optionparser.Register<unsigned long long>("-ulldata", "unsigned long long data", c.ulldata, "0x123456789abcdef1");
   optionparser.Register<bool>("-someflag", "first flag", c.bdata, "0");
   optionparser.Register<bool>("-otherflag", "second flag", (bool&)c.boolint, "1");
   optionparser.Register<char *>("-coption", "char * data", c.coption, NULL);

   cout << "Default: \n";
   optionparser.Print(stdout);

   optionparser.ParseCommandLine(argc, argv);

   cout << "Commandline Parse Results: \n";
   optionparser.Print(stdout);

   optionparser.ParseFile("test.config");
   cout << "File Parse Results: \n";
   optionparser.Print(stdout);
   cout << c.sdata << ' ' << c.idata << endl;

   return 0;
}

int cinterfacetest(int argc, char *argv[])
{
   testtype c;
   option_parser_t opp = option_parser_create();
   c.idata = 123;
   c.fdata = 3249586.333;
   c.sdata = string("haha");
    char *otherstr;

    option_parser_register(opp, "-idata", OPT_INT32, &c.idata, "integer data", "-456");
    option_parser_register(opp, "-fdata", OPT_FLOAT, &c.fdata, "floating point data", "0.001");
    option_parser_register(opp, "-sdata", OPT_CSTR, &otherstr, "first string data", "hellow");
    option_parser_register(opp, "-ulldata", OPT_UINT64, &c.ulldata, "unsigend long long data", "0x123456789abcdef1");
    option_parser_register(opp, "-someflag", OPT_BOOL, &c.bdata, "first flag", "0");
    option_parser_register(opp, "-otherflag", OPT_BOOL, &c.boolint, "second flag", "1");
    option_parser_register(opp, "-coption", OPT_CSTR, &c.coption, "char * data", NULL);

   printf("Default: \n");
   option_parser_print(opp, stdout);

   option_parser_cmdline(opp, argc, argv);

   printf("Commandline Parse Results: \n");
   option_parser_print(opp, stdout);

   option_parser_cfgfile(opp, "test.config");
   printf("File Parse Results: \n");
   option_parser_print(opp, stdout);
    printf("%s %d\n", otherstr, c.idata);

    option_parser_destroy(opp);

   return 0;
}

int main(int argc, char *argv[]) 
{
    cppinterfacetest(argc,argv);
    cinterfacetest(argc,argv);

    return 0;
}

#endif

