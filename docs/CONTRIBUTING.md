# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Get Started!

Ready to contribute? Here's how to set up `aicsimageio` for local development.

1.  Fork the `aicsimageio` repo on GitHub.

2.  Clone your fork locally:

    ```bash
    git clone https://{your_name_here}@github.com/aicsimageio.git
    ```

3.  Install the project in editable mode (and preferably in a virtual environment):

    ```bash
    cd aicsimageio/
    pip install -e .[dev]
    ```

4.  Download the test resources:

    ```bash
    python scripts/download_test_resources.py
    ```

    If you need to upload new resources please let a core maintainer know.

5.  Create a branch for local development:

    ```bash
    git checkout -b {your_development_type}/short-description
    ```

    Ex: feature/read-tiff-files or bugfix/handle-file-not-found<br>
    Now you can make your changes locally.

6.  When you're done making changes, check that your changes pass linting and
    tests, including testing other Python versions with make (!! Important !!):

    ```bash
    make build
    ```

    If you have any AWS credentials configured and would like to run the full
    remote IO test suite:

    ```bash
    make build-with-remote
    ```

7.  Commit your changes and push your branch to GitHub:

    ```bash
    git add .
    git commit -m "Resolves gh-###. Your detailed description of your changes."
    git push origin {your_development_type}/short-description
    ```

8.  Submit a pull request through the GitHub website.

## Benchmarking

If you are working on a patch that would change a base reader it is recommended
to run `asv` to benchmark how the change compares to the current release.

To do so simply run `asv` in the top level directory of this repo.
You can create a specific comparison by running `asv continuous branch_a branch_b`.

For more information on `asv` and full commands please see
[their documentation](https://asv.readthedocs.io/en/stable/).

## Deploying

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed.
Then run:

```bash
make prepare-release
git push
git push --tags
```

After all builds pass, GitHub Actions will automatically publish to PyPI.

**Note:** `make prepare-release` by default only bumps the patch number and
not a minor or major version.
