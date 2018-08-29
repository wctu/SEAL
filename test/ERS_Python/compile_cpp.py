from distutils.core import setup, Extension
import shutil

module = Extension('ERSModule',
                    sources = ['erspy.cpp', 'MERCCInput.cpp', 'MERCOutput.cpp', 'MERCDisjointSet.cpp', 'MERCFunctions.cpp', 'MERCLazyGreedy.cpp'])

setup(name = 'PackageName', 
      version = '1.0',
      description = 'This is a Python wrapper for ERS',
      ext_modules = [module])
      
shutil.copy2('./build/lib.linux-x86_64-2.7/ERSModule.so', 'ERSModule.so')
