Changelog
=========

v3.1.1 (2020-02-21)
-------------------

Fix
~~~
- Bugfix/make-aicsimage-serializable(#74) [JacksonMaxfield]
- Bugfix/return-none-cluster (#73) [Jamie Sherman]

Other
~~~~~
- Admin/auto-changelog (#75) [JacksonMaxfield]
- Admin/test-py38 (#76) [JacksonMaxfield]


v3.1.0 (2020-02-03)
-------------------

New
~~~
- Feature/use-dask (#63) [JacksonMaxfield]

Fix
~~~
- Fix pypi publish action. [Jackson Brown]
- Bugfix/auto-doc-gen (#70) [JacksonMaxfield]


v3.0.7 (2019-11-05)
-------------------
- Remove make clean command from make docs call (#49) [JacksonMaxfield]
- Populate_tiffdata should respect dimension order (#48) [toloudis]


v3.0.6 (2019-10-31)
-------------------

New
~~~
- Feature/physical pixel size (#43) [toloudis]

Fix
~~~
- Fix imread bug and allow AICSImage class to close its reader (#44)
  [toloudis]


v3.0.5 (2019-10-30)
-------------------
- Clean up from PR comments. [Daniel Toloudis]
- Add get_channel_names to AICSImage class. [Daniel Toloudis]


v3.0.4 (2019-10-28)
-------------------
- Add size getters to the AICSImage class (#38) [toloudis]


v3.0.3 (2019-10-25)
-------------------

Fix
~~~
- Fix linting. [Daniel Toloudis]
- Fix png writer and tests. [Daniel Toloudis]
- Fix linter. [Daniel Toloudis]
- Fix png writer and tests. [Daniel Toloudis]

Other
~~~~~
- Remove patch coverage check (#36) [JacksonMaxfield]
- Pull request code review revisions. [Dan Toloudis]
- Revert "fix png writer and tests" [Daniel Toloudis]
- Use old default for dimension_order so that existing code does not
  break. [Daniel Toloudis]
- Add a unit test for dimension_order and refactor test_ome_tiff_writer.
  [Daniel Toloudis]
- Allow dimension order in ome-tiff writer. [Daniel Toloudis]
- Remove accidentally added file. [Daniel Toloudis]
- Remove CRON from doc build workflow. [Jackson Brown]
- Remove double builds from github actions. [Jackson Brown]
- Wrap CRON string in quotes. [Jackson Brown]
- Update CRON strings. [Jackson Brown]
- Do not build documentation for tests module. [Jackson Brown]
- Update makefile to remove all generated rst's on doc gen. [Jackson
  Brown]
- Update czireader import so that it doesn't fail on etree. [Jackson
  Brown]
- Move documentation badge to before codecov. [JacksonMaxfield]
- Update readme to have doc badge. [JacksonMaxfield]
- Add doc generation workflow. [JacksonMaxfield]
- Add required documentation files and update requirements.
  [JacksonMaxfield]
- Update task version pins to point at master / latest.
  [JacksonMaxfield]


v3.0.2 (2019-10-11)
-------------------
- Pull in feedback from team. [Jackson Brown]
- Update README to include known_dim functionality. [Jackson Brown]
- Add test for invalid dim names. [Jackson Brown]
- Resolves [gh-22], allow passing of known dim order to AICSImage.
  [Jackson Brown]
- Resolves [gh-23], use OME-Tiff metadata to parse dim sizes and order.
  [Jackson Brown]


v3.0.1 (2019-10-04)
-------------------
- Label Quickstart code block as python. [Jackson Brown]
- Update setup.cfg to properly bumpversion. [Jackson Brown]
- Rename build workflow to build master. [Jackson Brown]
- Update to new cookiecutter gh templates after matts feedback. [Jackson
  Brown]
- Remove cov report html from actions as not needed on remote. [Jackson
  Brown]
- Add PR to test and lint action triggers. [Jackson Brown]
- Remove references to quilt3distribute that were copied over. [Jackson
  Brown]
- Update CI/CD, README badge, local developement, and contributing docs.
  [Jackson Brown]
- CODE_OF_CONDUCT.md. [Jamie Sherman]
