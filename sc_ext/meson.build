module = python.extension_module(
  'sc_ext',          # module_name
  ['set_cover.cpp'], # list of sources
  install: true,
  dependencies: [
    python.dependency() # Add Python.h as dependency
  ],
  subdir: 'sc_ext', # Folder relative to site-packages to install to
)
# Pure Python sources
python_sources = ['__init__.py']

# Install pure Python
python.install_sources(
  python_sources,
  pure: false, # install native extensions alongside pure python modules
  subdir: 'sc_ext' # Folder relative to site-packages to install to
)
# python3 -m build . --skip-dependency-check --no-isolation