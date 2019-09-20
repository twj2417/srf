import os

from setuptools import setup, find_packages

MAJOR = 0
MINOR = 12
MINOR_SUB = 6
MICRO = 1  # 0 for alpha, 1 for beta, 2 for release candicate, 3 for release


def get_version(major, minor, micro, minor_sub):
    from time import time
    from datetime import datetime
    timestamp = int(time())
    short_version = str(major) + '.' + str(minor) + '.' + str(minor_sub) + '.' + str(micro)
    full_version = short_version + '.' + str(timestamp)
    time_string = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    return short_version, full_version, time_string


SHORT_VERSION, FULL_VERSION, TIME_STRING = get_version(MAJOR, MINOR, MINOR_SUB, MICRO)


def write_version_py(filename = 'srfnef/version.py'):
    cnt = """
# THIS FILE IS GENERATED FROM SRFNEF SETUP.PY
# 
short_version = '%(short_version)s'
full_version = '%(full_version)s'
generated_time = '%(time_string)s'
    
    """
    with open(filename, 'w') as fin:
        fin.write(cnt % {'short_version': SHORT_VERSION,
                         'full_version': FULL_VERSION,
                         'time_string': TIME_STRING})


def run_pytype(out_path = None):
    if out_path is None:
        out_path = os.path.abspath('./.pytype')


write_version_py()

setup(name = 'srfnef',
      version = FULL_VERSION,
      py_modules = ['srfnef'],
      description = 'Scalable Reconstruction Framework -- Not Enough Functions',
      author = 'Minghao Guo',
      author_email = 'mh.guo0111@gmail.com',
      license = 'Apache',
      # packages = ['srfnef'],
      packages = find_packages(),
      install_requires = [
          'scipy',
          'matplotlib',
          'h5py',
          'click',
          'numpy',
          'tqdm',
          'numba',
          # 'deepdish==0.3.6',
          'attr',
          'attrs',
          'pypandoc',
          'dxl-learn'
      ],
      zip_safe = False,
      entry_points = {'console_scripts': [
          'srfnef.extract_listmode = srfnef.app.app_extract_listmode:extract_listmode',
          'srfnef.recon_simple = srfnef.app.app_recon_simple:recon_simple',
          'srfnef.recon_full = srfnef.app.app_recon_full:recon_full',
          'srfnef.recon_with_scatter = srfnef.app.app_recon_full:recon_with_scatter',
          'srfnef.recon_full_cylindrical = srfnef.app.app_recon_full_cylindrical:recon_full',
          'srfnef.test_all = srfnef.app.app_test_full:test_full',
          'srfnef.test_with_scatter = srfnef.app.app_test_full:test_with_scatter',
          'srfnef.test_all_tof = srfnef.app.app_test_full_tof:test_full_tof',
          'srfnef.test_with_scatter_tof = srfnef.app.app_test_full_tof:test_with_scatter_tof']
      },
      )
