# AICS Image library
The aicsimageio package is designed to provide an easy interface with CZI, OME-TIFF, and PNG file formats.

To install:

(1) if you are already set up to install from AICS artifactory, (see http://confluence.corp.alleninstitute.org/display/SF/Using+Artifactory) use:
    >>> pip install aicsimageio
    or
    >>> pip install aicsimageio -i https://artifactory.corp.alleninstitute.org/artifactory/api/pypi/pypi-virtual/simple

OR 
(2) to install directly from this source repo:
    >>> pip install .


## Level of Support
We are not currently supporting this code for external use, but simply releasing it to the community AS IS. It is used for within our organization. We are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.

## Project Structure
Note the important aspects of the file structure for the python package in the segment below. 
This is discussed in more detail in the docs [STRUCTURE.md](STRUCTURE.md).
 
```
repo/
|-- modulename/     # E.g. aicsimageio in this example
|   |-- __init__.py
|   |-- module_A.py
|   |-- module_B.py
|   |
|   |-- bin/
|   |   |-- __init__.py
|   |   |-- cli1.py # Module with a main() for entrypoint
|   |   |-- cli2.py # Module with a main() for entrypoint
|   |
|   |-- tests/
|      |-- __init__.py
|      |-- test_A.py
|      |-- test_B.py
|
|-- setup.py
|-- setup.cfg
|-- README.md
|-- requirements.txt
|-- docs/
```

In addition to the above we have the following for build management and for legal purposes.

```
repo/
|-- CONTRIBUTING.md
|-- LICENSE.txt
|-- build.gradle
|-- settings.gradle
|-- gradlew
|-- gradlew.bat
|-- gradle/
```



## Setup, Test, Build and Publish

We use gradle to manage the builds. The build files are discussed along with the directory structure in [STRUCTURE.md](STRUCTURE.md). You must have Java installed and the `JAVA_HOME` environment set for this to work.

#### Setup

Before running any tests or builds you should setup the virtual environment for it. (Currently this also attempts to build the project.)

```bash
./gradlew setupVirtualEnv 
```

#### Dev Setup

This is the same as setUpVirtualEnv, but it will install the packages required for tests.

```bash
./gradlew installDevDependencies 
```

#### Test

This will just run the tests.

```bash
./gradlew test
```

Note that the above creates a Jenkins compatible report at ./build/test_report.xml 

#### Local Build

The following will run the tests, build the package locally, and store the wheel in the `<repo>/dist/`.
```bash
./gradlew build
```
You can install it to a virtual environment of your choosing using the wheel e.g.
```
pip install ./dist/aicsimageio-0.0.1-py2-py3-none-any.whl
```

#### Publish

This will run the unit tests, and then build and publish to artifactory. Typically only the build system should be allowed to publish to the PyPi repositories. 
```bash
./gradlew publish
```
The publish tasks puts the wheel into the snapshot repository. Typically once the package is properly tested you then 'promote' it to the production repository. As a developer your machine maybe setup to pull from both the production and the snapshot Python repositories in which case you can download the pre-release package. To install from artifactory you should be able to just run the following, updating to the latest package.
```bash
pip install aicsimageio -U
```


#### Publication configuration

To allow publication on Jenkins the  publication configuration file `~/.pypirc` should exist in the Jenkins user home directory. (If you need to test publish capabilities, it should be on your machine too.). It should look like this below.
```
[distutils]
index-servers =
    snapshot-local

[snapshot-local]
repository: https://artifactory.corp.alleninstitute.org/artifactory/api/pypi/pypi-snapshot-local
username: <username>
password: <password or artifactory key>
```

#### Pip configuration for downloading

To pull from both the internal artifact repository as well as the global PyPi repository you need to have your machine properly configured. **Talk to someone in SW (DevOps or Sys. Admin.) to ensure the system is configured correctly using the ansible scripts**.

You need to have the **AICS certificates installed**. Additionally you need the `pip.conf` file setup. 
The file should be located at the following location.

| Platform | Location |
| --- | --- |
| Linux | `~/.config/pip/pip.conf` |
| Windows | `%APPDATA%\pip\pip.ini` |
| macOS | `$HOME/Library/Application Support/pip/pip.conf` |
|||

The file should contain the following
```
[global]
cert = /etc/ssl/certs
index-url = https://artifactory.corp.alleninstitute.org/artifactory/api/pypi/pypi-virtual/simple
```

To allow access to the snapshot repository to work with the development versions of packages you can add the following line to the `pip.conf` file.
```
extra-index-url = https://artifactory.corp.alleninstitute.org/artifactory/api/pypi/pypi-snapshot-local/simple
```

## References
- http://python-packaging.readthedocs.io/en/latest/index.html
