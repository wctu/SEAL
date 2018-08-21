from distutils.core import setup, Extension
import shutil

module = Extension('EvalSPModule', sources = ['eval_superpixel.cpp'])

setup(name = 'PackageName', 
      version = '1.0',
      description = 'This is a Python wrapper for eval_superpixel.cpp',
      ext_modules = [module])
      
shutil.copy2('./build/lib.linux-x86_64-2.7/EvalSPModule.so', 'EvalSPModule.so')
