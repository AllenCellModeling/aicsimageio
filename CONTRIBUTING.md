# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Get Started!
Ready to contribute? Here's how to set up `aicsimageio` for local development.

1. Fork the `aicsimageio` repo on GitHub.

1. Clone your fork locally:

    ```bash
    git clone git@github.com:{your_name_here}/aicsimageio.git
    ```

1. Install the project in editable mode. (It is also recommended to work in a virtualenv or anaconda environment):

    ```bash
    cd aicsimageio/
    pip install -e .[dev]
    ```

1. Download the test resources:

    ```bash
    python scripts/download_test_resources.py
    ```

    If you need to upload new resources please let a core maintainer know.

1. Create a branch for local development:

    ```bash
    git checkout -b {your_development_type}/short-description
    ```

    Ex: feature/read-tiff-files or bugfix/handle-file-not-found<br>
    Now you can make your changes locally.

1. When you're done making changes, check that your changes pass linting and
   tests, including testing other Python versions with make:

    ```bash
    make build
    ```

1. Commit your changes and push your branch to GitHub:

    ```bash
    git add .
    git commit -m "Resolves gh-###. Your detailed description of your changes."
    git push origin {your_development_type}/short-description
    ```

1. Submit a pull request through the GitHub website.

## Deploying

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed.
Then run:

```bash
make prepare-release
git push
git push --tags
git branch -D stable
git checkout -b stable
git push --set-upstream origin stable -f
```

This will release a new package version on Git + GitHub and publish to PyPI.
