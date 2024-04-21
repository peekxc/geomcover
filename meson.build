# https://mesonbuild.com/
project('sc_ext', 'cpp', version: '0.1')

# https://mesonbuild.com/Python-module.html
pymod = import('python')
python = pymod.find_installation('python3', modules:['numpy'], required:true) # modules:['numpy'] needed for ubuntu
pydep = python.dependency(version:'>=3.8.0', components:['Interpreter', 'Development', 'NumPy'])

subdir('sc_ext') # Evaluate meson.build in greetpkg/

## Summary:
## mesonpep517 is just too buggy and too light. Find a way to build with meson directly via CLI 
## and then link extensions w/ poetry somehow w/ setuptools 