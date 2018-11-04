# Build Info

- [Introduction](#intro)
- [Build Tasks](#build_tasks)
  - [Testing Console Scripts](#console_scripts)
  - [Working On Multiple Python Repos](#multiple_repos)
- [Continuous Integration (CI) and Version Management](#ci)

## Introduction<a name="intro"></a>

This document discusses the build process and the basic layout of the project.

The build process is setup so that you can develop in one of two ways
- using only python, or
- using gradle (which calls the python setup anyway).

We use the gradle setup in conjunction with our CI process and Jenkins since it encapsulates 
the build tool chain.

In either case before you start development you will need to do the following.

1. Clean the folder - this is useful if you are reseting the workspace.
2. You will need a virtual environment for the development process.
    - Python-Only: You will need to manually create one. We typically keep 
    it under the project directory
        ```bash
        export venvdir=~/venvs/buildproject
        virtualenv -p python3 ${venvdir}
        source ${venvdir}/bin/activate
        ```
        You will need to remember to activate the environment before you run
        any of the subsequents steps.
    - Gradle: When you run the any of the gradle build, test or lint tasks, gradle 
    will automatically ensure you have a virtual environment. The default location is 
    `<repo_dir>/venv/x3`. It also creates the symlink 
    `<repo_dir>/activate -> <repo_dir>/venv/x3/bin/activate` to simplify access.
3. Install the dependencies into the environment (see the "Build Tasks" table below). 
Typically you will want to use (A) the standard dev setup, or (B) the interactive setup. 
The second one if you setup your project with tooling specified in the 
`interactive_dev_deps` group in `setup.py`. Use it to install modules like Jupyter.
Note that for continuous integration, and manual version management you will need (C).

## Build Tasks<a name="build_tasks"></a>

These are a list of common tasks necessary for building and testing. Note that the 
gradle commands can be abbreviated by initials - i.e. `./gradlew <abbrev>` using 
the abbreviations shown below.

|Task info|Gradle|Python|
|:---|:---|:---|
|**Clean workspace**<br/>This deletes all generated<br/>files and folders|`./gradlew cleanAll`<br/>Abbrev: `cA`|`rm -rf dist/ build/ .eggs/ \`<br/>`.pytest_cache ${project.name}.egg-info`|
|**Setup standard dev (A)**|`./gradlew installDependencies`<br/>Abbrev: `iD`|`pip install -e .[test_group] -e .[lint_group]`|
|**Setup interactive dev (B)**<br/>support e.g. Jupyter|`./gradlew installInteractiveDevDependencies`<br/>Abbrev: `iIDD`|`pip install -e .[test_group] -e .[lint_group]\`<br/>`-e .[interactive_dev_group]`|
|**Setup CI environment (C)**<br/>Auto installs (A)|`./gradlew installCIDependencies`<br/>Short: `./gradlew iCID`| - |
|**Run linter**|`./gradlew lint`|`flake8 --count --exit-zero {{ cookiecutter.python_module_name }}`|
|**Run tests**|`./gradlew testPackage`<br/>Abbrev: `tP`, `test`|`python setup.py test`|
|**Build wheel**|`./gradlew build`<br/>This always runs lint and test.|`python setup.py bdist_wheel`|
|**List all gradle tasks**|`./gradlew tasks`||
|||

### Testing Console Scripts<a name="console_scripts"></a>

To run your console scripts on the command line you will want the virtual environment
activated. Additionally for development, you will need to ensure your module is 
source-installed in editable mode into the environment.

- Python: Activate the virtual env and run `pip install -e .` in the project directory.
- Gradle: If you run the `test` or `build` task once, will automatically be source-installed 
into the default environment. Note that you will also have the `activate` symlink in the project 
directory to activate the environment.

### Working On Multiple Python Repos<a name="multiple_repos"></a> 

There are scenarios where you want to work on multiple python modules simultaneously (each from a separate repo), 
e.g., one module depends on another and you editing both. In that scenario you will have to pick one virtual environment
and do source install into it from all the relevant module repos.

Assuming we have project A and B with modules `mod_a` and `mod_b`, where `mod_b` depends on `mod_a`. 
We want to work on both together. Let's use the virtual environment created by gradle for B. To setup the enviroment 
for development on both A and B, I do the following.
```
# We are in B's workspace. First setup the virtual environment
$B> ./gradlew cleanAll installDependencies


# Frist activate the environment. Not that gradle creates a convenient symlink to activate the environment
$B> . activate

# If you ran   ./gradlew build   or   ./gradlew test
# it will already source install the module for B. For the example we do it again anway.
(x3)$B> pip install -e .

# Now cd to the director for A, and install it as source
(x3)$B> cd ../A
(x3)$A> pip install -e .
```

Now code edits in either A or B will be available in the "installed" setup. Note that if you add or remove console 
script files (the ones mentioned in `entry_points` in `setup.py`), you will need to run the source code install 
step again.

## Continuous Integration (CI) and Version Management<a name="ci"></a> 

The version management happens in the continuous integration process. This relies on using semantic versioning 
combined with specifications in PEP-440.

The development version is of the form `<major>.<minor>.<patch>.dev<devbuild>`, where each component is an integer. When changes are merged into the master branch, the CI process will increment the `<devbuild>`: `X.Y.Z.devN` -> `X.Y.Z.devN+1`.

For a release, the CI process will drop the last component, creating version `<major>.<minor>.<patch>`: `X.Y.Z.devN` -> `X.Y.Z`. 

Upon publishing and tagging after a release, it will increment the `<patch>` number: `X.Y.Z` -> `X.Y.Z+1.dev0`. 

If your next release requires an increase in the `<major>` or `<minor>` versions, You can use the gradle tasks to increment the
version using the `bumpVersionPostRelease` task with an additional parameter
```bash
# To increment the major part
./gradlew bumpVersionPostRelease -PbumpPartOverride=major

# To increment the minor part
./gradlew bumpVersionPostRelease -PbumpPartOverride=minor
```

Note that the gradlew tasks calls the Python `bumpversion` module as part of the task, but it also provides a number of checks to avoid complications from certain `bumpversion` behavior.



