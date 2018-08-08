# File/Folder Structure

We are using the folder structure of this project as a guide for any new python packages we develop.

## Top Level Structure

The structure has a number of top level files

- `setup.py`: This is the standard file used to test, build and upload python packages, based on the setuptools package. We discuss this a bit more in different segments below.
- `setup.cfg`: This is a `.ini` format file that holds additional options for the package setup process. The file in this repository has options we expect to use in most packages we develop.
  - We should be aiming to developing universal wheels in most cases, i.e., python packages that will work in both Python 2 and 3, and with no native code requirements. As such we have `universion=1` in the `[bdist_wheel]` group. 
  - For testing we use `pytest`. In the event we change to a different tool, we can keep things consistent by modifying the alias and the tool options in the `setup.cfg` file.
- `README.md`: This is the standard README file for top level documentation.
- `requirements.txt`: This file is typically used to host the dependencies used by the package. In many cases it is created by saving the output of `pip freeze` although this brings along every single package install - not just dependencies. To reduce redudant declaration of dependencies, we only add a `.` to the file and declare the dependencies in `setup.py` in the `install_requires` option. E.g.,
  ```python
  setup(...,
      install_requires=[
          "json5>=0.5.0",
      ], 
  ...)
  ```
## Package Folder Structure

### Modules and submodules
The package directory (e.g. `aicsimageio` in this example repo) contains an `__init__.py` file to signify it is a module. Any subdirectory that will be treated as a submodule for packaging or testing should have a `__init__.py` file. You can create module (`*.py`) files in this directory or in subdirectories that will be treated as submodules.

### Why is `bin` a submodule?
The `bin/` directory under the top level is a special submodule. This hosts python files that are meant to be run as command line tools. We could have them anywhere in the package heirarchy, but this ensure proper organization. If necessary you could create associated modules and submodules in here purely for CLI management. 
- The actual command line scripts will be automatically generated when a user pip installs the generated wheel. It will take care of the platform behaviour generating appropriate scripts for Windows, Linux and Mac. The scripts are install in the users' path, or the corresponding location in any virtual environment they use.
- For the scripts to generated during install, the entrypoint must be specified in `setup.py` in the `entry_points` option. E.g.
  ```python
  setup(...,
      entry_points={
            "console_scripts": [
                "time-xfer=aicsimageio.bin.xfer_time:main",
                "timing-demo=aicsimageio.bin.timing_demo:main"
            ]
          },
  ...)
  ```
- The python modules in bin, e.g. `xfer_time.py` should have a function to identify the entrypoint to be called. We choose to call it `main()`. Inside of the main you access  `sys.argv` the same way you would in a regular command line script. To allow for testing during development the file should also include a segment to make the module directly callable in your development environment. E.g.
  ```python
  ...

  def main(**kwargs):
    """ The entrypoint """
    # Use sys.argv to get command line arguments in here or in functions/objects 
    # called from here

  # Make the script directly callable from the command line for development.
  if __name__ == `__main__`:
    main()
  ```

### Why do we have `tests` submodules
There are multiple camps in regards to how to layout tests in a python package. They all have their pros and cons. We chose to incorporate tests as in submodules called `tests`. Note that every actual submodule or module could have its own `tests` submodule. This simplifies autodiscovery of tests, and simplifies the development package access. This should make it easier for anyway using the 

## Testing

For testing we currently chose to use `pytest` since this can create a Jenkins compatible report and also run unit tests based on the `unittest` module in standard distribution. To allow this we declare the test packages in `setup.py`.
  ```python
  setup(...
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
  ...)
  ```
  Note the option declaration in `setup.cfg` discuss earlier.

## Build files

We use gradle for managing builds in continuous integration on Jenkins. This will also work from the command line on your own machine as long as you have Java installed and the `JAVA_HOME` environment variable setup correctly. The project build relies on two files.
- `settings.gradle`: The name of the project and hence the python package is defined here. This ensures that the project name is independent of the repository name.
- `build.gradle`: This is the primary file with `tasks`, i.e., instructions on how to build the project. Note that these were shown in the [README.md](README.md).

### Versioning
The version of the package should be declared only in one place to ensure consistency. This is done in `build.gradle` in the line declaring `project.version`. We use semantic versioning here.

Note that when you build with gradle, it will first create the version file `<packagedir>/version.py` and then call `setup.py`. This way the version becomes a part of the code that can be queried by any client of the package. Additionally it is made available to the `setup.py` package during the build. This version will be used to tag a released package.


