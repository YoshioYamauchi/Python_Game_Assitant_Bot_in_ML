

# Run this command:
# python setup01.py build_ext --inplace

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import os
import imp



if os.path.exists('./cameron_v2/cython_utils/nms.c'): os.remove('./cameron_v2/cython_utils/nms.c')
if os.path.exists('./cameron_v2/cython_utils/nms.so'): os.remove('./cameron_v2/cython_utils/nms.so')
if os.path.exists('./cameron_v2/cython_utils/cy_yolo_findboxes.c'): os.remove('./cameron_v2/cython_utils/cy_yolo_findboxes.c')
if os.path.exists('./cython_utils/cy_yolo_findboxes.so'): os.remove('./cameron_v2/cython_utils/cy_yolo_findboxes.so')
if os.path.exists('./cameron_v2/cython_utils/cy_yolo2_findboxes.c'): os.remove('./cameron_v2/cython_utils/cy_yolo2_findboxes.c')
if os.path.exists('./cameron_v2/cython_utils/cy_yolo2_findboxes.so'): os.remove('./cameron_v2/cython_utils/cy_yolo2_findboxes.so')
if os.path.exists('./cameron_v2/cython_utils/cy_yolo2_findboxes02.c'): os.remove('./cameron_v2/cython_utils/cy_yolo2_findboxes02.c')
if os.path.exists('./cameron_v2/cython_utils/cy_yolo2_findboxes02.so'): os.remove('./cameron_v2/cython_utils/cy_yolo2_findboxes02.so')



# VERSION = imp.load_source('version', os.path.join('.', 'darkflow', 'version.py'))
# VERSION = VERSION.__version__
#
#

ext_modules=[
    # Extension("cameron_v2.cython_utils.nms",
    #     sources=["./cameron_v2/cython_utils/nms.pyx"],
    #     libraries=["m"], # Unix-like specific
    #     include_dirs=[numpy.get_include()]
    # ),
    # Extension("cameron_v2.cython_utils.cy_yolo2_findboxes",
    #     sources=["./cameron_v2/cython_utils/cy_yolo2_findboxes.pyx"],
    #     libraries=["m"], # Unix-like specific
    #     include_dirs=[numpy.get_include()]
    # ),
    # Extension("cameron_v2.cython_utils.cy_yolo_findboxes",
    #     sources=["./cameron_v2/cython_utils/cy_yolo_findboxes.pyx"],
    #     libraries=["m"], # Unix-like specific
    #     include_dirs=[numpy.get_include()]
    # ),
    Extension("cameron_v2.cython_utils.cy_yolo2_findboxes02",
        sources=["./cameron_v2/cython_utils/cy_yolo2_findboxes02.pyx"],
        libraries=["m"], # Unix-like specific
        include_dirs=[numpy.get_include()]
    ),
    Extension("cameron_v2.cython_utils.prediction_utils_cy01",
        sources=["./cameron_v2/cython_utils/prediction_utils_cy01.pyx"],
        libraries=["m"],
        include_dirs=[numpy.get_include()]
        )
]



# The GNU General Public License (GPL) is a wildely used free software license,
# which guaranteees end users the freedom to run, study, share and modify the
# software.
#
# setup(
#     version=VERSION,
#     name='darkflow',
#     description='Darkflow',
#     license='GPLv3',
#     url='https://github.com/thtrieu/darkflow',
#     packages = find_packages(),
#     scripts = ['flow'],
#     ext_modules = cythonize(ext_modules)
# )
setup(name ='cameron_v2',
      ext_modules=cythonize(ext_modules),
      packages=find_packages(),
      author='spparkle',
      author_email='ckv13192@ict.nitech.ac.jp',
      description='cheat with macine learning',
      url='https://medium.com/@y1017c121y',
      license='GPLv3'
      )
