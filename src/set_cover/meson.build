python_sources = [
  '__init__.py', 
  '__version__.py', 
  'loaders.py',
  'csgraph.py',
  'covers.py', 
  'wset_cover.py', 
  'plotting.py'
]

## Extension modules
# module = py.extension_module(
#   '_cover',      
#   sources: ['_cover.cpp'],
#   include_directories: includes,
#   install: true, 
#   cpp_args: _cpp_args
# )

## Print install directory
message('Source install dir = '+py.get_install_dir())
py.install_sources(
  python_sources,
  subdir: pkg_name, 
  pure: false       # Will be installed next to binaries
)