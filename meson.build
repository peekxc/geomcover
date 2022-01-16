# https://mesonbuild.com/
project(
  'sc_ext', 
  'cpp',
  version: '0.1'
)

# https://mesonbuild.com/Python-module.html
pymod = import('python')
python = pymod.find_installation(
  required: true,
)
subdir('sc_ext') # Evaluate meson.build in greetpkg/