# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Get Started!
Ready to contribute? Here's how to set up `aicsimageio` for local development.

1. Fork the `aicsimageio` repo on GitHub.
2. Clone your fork locally:

    $ git clone git@github.com:{your_name_here}/aicsimageio.git

3. Install the project in editable mode. (It is also recommended to work in a virtualenv or anaconda environment):

    $ cd aicsimageio/
    $ pip install -e .[dev]

4. Create a branch for local development:

    $ git checkout -b {your_development_type}/short-description

    Ex: feature/read-tiff-files or bugfix/handle-file-not-found

    Now you can make your changes locally.

5. When you're done making changes, check that your changes pass linting and
   tests, including testing other Python versions with tox:

    $ tox

6. Commit your changes and push your branch to GitHub:

    $ git add .
    $ git commit -m "Resolves gh-###. Your detailed description of your changes."
    $ git push origin {your_development_type}/short-description

7. Submit a pull request through the GitHub website.

## Deploying

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed.
Then run:

$ bumpversion patch # possible: major / minor / patch
$ git push
$ git push --tags

Travis will then deploy to PyPI if tests pass.


## Allen Institute Contribution Agreement
This document describes the terms under which you may make “Contributions” —
which may include without limitation, software additions, revisions, bug fixes, configuration changes,
documentation, or any other materials — to any of the projects owned or managed by the Allen Institute.
If you have questions about these terms, please contact us at terms@alleninstitute.org.

You certify that:

• Your Contributions are either:

1. Created in whole or in part by you and you have the right to submit them under the designated license
   (described below); or
2. Based upon previous work that, to the best of your knowledge, is covered under an appropriate
   open source license and you have the right under that license to submit that work with modifications,
   whether created in whole or in part by you, under the designated license; or
3. Provided directly to you by some other person who certified (1) or (2) and you have not modified them.

• You are granting your Contributions to the Allen Institute under the terms of the [2-Clause BSD license](https://opensource.org/licenses/BSD-2-Clause)
  (the “designated license”).

• You understand and agree that the Allen Institute projects and your Contributions are public and that
  a record of the Contributions (including all metadata and personal information you submit with them) is
  maintained indefinitely and may be redistributed consistent with the Allen Institute’s mission and the
  2-Clause BSD license.
