import os 
import sysconfig
import distutils.sysconfig
from typing import Any, Dict
from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

base_path = os.path.dirname(__file__)
extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-std=c++17", "-Wall", "-Wextra", "-O2"]

flags = distutils.sysconfig.get_config_var("CFLAGS")
print(f"COMPILER FLAGS: { str(flags) }")

ext_modules = [
  Pybind11Extension(
    'set_cover_ext', 
    sources = ['src/set_cover/sc_ext/set_cover.cpp'], 
    # include_dirs=['/Users/mpiekenbrock/diameter/extern/pybind11/include'], 
    extra_compile_args=extra_compile_args,
    language='c++17', 
    cxx_std=1
  )
]

def build(setup_kwargs: Dict[str, Any]) -> None:
  setup_kwargs.update({
    "name": "set_cover",
    "version": "0.1.0",
    "ext_modules": ext_modules,
    "cmdclass": dict(build_ext=build_ext),
    "zip_safe": False, # ensures platform-specific wheel is created
    "packages": find_packages()
  })

# For python develop: pip install --editable .
# For c++ develop: python3 -m build --no-isolation --wheel --skip-dependency-check