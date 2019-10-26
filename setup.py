from setuptools import find_packages, setup

PACKAGE_NAME = 'oldaicsimageio'


"""
Notes:
We get the constants MODULE_VERSION from
See (3) in following link to read about versions from a single source
https://packaging.python.org/guides/single-sourcing-package-version/#single-sourcing-the-version
"""

MODULE_VERSION = ""
exec(open(PACKAGE_NAME + "/version.py").read())

def readme():
    with open('README.md') as f:
        return f.read()

test_deps = ['pytest', 'pytest-cov', 'pytest-raises']
lint_deps = ['flake8']
interactive_dev_deps = []
extras = {
      'test_group': test_deps,
      'lint_group': lint_deps,
      'interactive_dev_group': interactive_dev_deps
}

setup(name=PACKAGE_NAME,
      version=MODULE_VERSION,
      description='A generalized scientific image processing module from the Allen Institute for Cell Science.',
      long_description=readme(),
      long_description_content_type="text/markdown",
      author='AICS',
      author_email='!AICS_SW@alleninstitute.org',
      license='Allen Institute Software License',
      packages=find_packages(exclude=['tests', '*.tests', '*.tests.*']),
      entry_points={
          "console_scripts": [
          ]
      },
      install_requires=[
            'imageio>=2.3.0',
            'numpy>=1.14.5',
            'Pillow>=5.2.0',
            'scipy>=1.1.0',
            'matplotlib>=2.2.2', # get >=2.2.3 when available, because of https://github.com/matplotlib/matplotlib/pull/10867
            'scikit-image>=0.14.0',
            'tifffile==0.15.1'
      ],

      # For test setup. This will allow JUnit XML output for Jenkins
      setup_requires=['pytest-runner'],
      tests_require=test_deps,

      extras_require=extras,
      zip_safe=False
      )
