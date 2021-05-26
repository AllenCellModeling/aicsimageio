Changelog
=========

(unreleased)
------------

New
~~~
- feature/mosaic-tile-single-position-request-and-docs  (`#229
  <https://github.com/AllenCellModeling/aicsimageio/pull/229>`_)
  [Jackson Maxfield Brown]
- feature/set-known-coords  (`#224
  <https://github.com/AllenCellModeling/aicsimageio/pull/224>`_)
  [Jackson Maxfield Brown]
- feature/aicsimage-save  (`#215
  <https://github.com/AllenCellModeling/aicsimageio/pull/215>`_)
  [Jackson Maxfield Brown]
- feature/lif-reader  (`#212
  <https://github.com/AllenCellModeling/aicsimageio/pull/212>`_)
  [Jackson Maxfield Brown]
- feature/ome-tiff-writer-4  (`#211
  <https://github.com/AllenCellModeling/aicsimageio/pull/211>`_)
  [JacksonMaxfield, toloudis]
- feature/array-like-reader  (`#197
  <https://github.com/AllenCellModeling/aicsimageio/pull/197>`_)
  [Jackson Maxfield Brown]
- feature/writers  (`#198
  <https://github.com/AllenCellModeling/aicsimageio/pull/198>`_)
  [Jackson Maxfield Brown]
- feature/add-aicsimage-obj  (`#185
  <https://github.com/AllenCellModeling/aicsimageio/pull/185>`_)
  [Jackson Maxfield Brown]
- feature/optimize-tiff-reader-and-add-benchmarks  (`#183
  <https://github.com/AllenCellModeling/aicsimageio/pull/183>`_)
  [Jackson Maxfield Brown]
- feature/ome-tiff-reader  (`#176
  <https://github.com/AllenCellModeling/aicsimageio/pull/176>`_)
  [Jackson Maxfield Brown]
- feature/add-tiff-reader  (`#160
  <https://github.com/AllenCellModeling/aicsimageio/pull/160>`_)
  [Jackson Maxfield Brown]
- feature/add-default-reader  (`#157
  <https://github.com/AllenCellModeling/aicsimageio/pull/157>`_)
  [Jackson Maxfield Brown]
- feature/add-timeseries-writer  (`#137
  <https://github.com/AllenCellModeling/aicsimageio/pull/137>`_)
  [Jackson Maxfield Brown]
- feature/add-rgb-writer  (`#134
  <https://github.com/AllenCellModeling/aicsimageio/pull/134>`_)
  [JacksonMaxfield]
- feature/centralize-reader-tests  (`#135
  <https://github.com/AllenCellModeling/aicsimageio/pull/135>`_)
  [JacksonMaxfield]
- feature/writer-base-class-proposal  (`#98
  <https://github.com/AllenCellModeling/aicsimageio/pull/98>`_)
  [JacksonMaxfield]
- feature/deprecate-context-manager-cluster-spawning  (`#97
  <https://github.com/AllenCellModeling/aicsimageio/pull/97>`_)
  [JacksonMaxfield]
- feature/deprecate-napari-functionality  (`#96
  <https://github.com/AllenCellModeling/aicsimageio/pull/96>`_)
  [JacksonMaxfield]

Fix
~~~
- bugfix/always-use-synch-for-tiff-zarr-compute  (`#235
  <https://github.com/AllenCellModeling/aicsimageio/pull/235>`_)
  [Jackson Maxfield Brown]
- bugfix/setup-coords-for-stitched-lifs  (`#234
  <https://github.com/AllenCellModeling/aicsimageio/pull/234>`_)
  [Jackson Maxfield Brown]
- bugfix/remove-unused-lif-code  (`#226
  <https://github.com/AllenCellModeling/aicsimageio/pull/226>`_)
  [Jackson Maxfield Brown]
- bugfix/catch-all-errors-in-ome-tiff  (`#207
  <https://github.com/AllenCellModeling/aicsimageio/pull/207>`_)
  [Jackson Maxfield Brown]
- bugfix/provide-correct-indices-to-data-select-in-tiff-chunked-reads
  (`#201 <https://github.com/AllenCellModeling/aicsimageio/pull/201>`_)
  [Jackson Maxfield Brown]
- Fix benchmarks links and remove old files [JacksonMaxfield]
- bugfix/adopt-samples-as-rgb-default  (`#165
  <https://github.com/AllenCellModeling/aicsimageio/pull/165>`_)
  [Jackson Maxfield Brown]
- Fix guess tiff dims, lint, and format [JacksonMaxfield]

Other
~~~~~
- admin/copy-v3 docs-to-static  (`#233
  <https://github.com/AllenCellModeling/aicsimageio/pull/233>`_)
  [Jackson Maxfield Brown]
- admin/split-reader-deps  (`#221
  <https://github.com/AllenCellModeling/aicsimageio/pull/221>`_)
  [Jackson Maxfield Brown]
- admin/doc-updates  (`#216
  <https://github.com/AllenCellModeling/aicsimageio/pull/216>`_)
  [Jackson Maxfield Brown]
- admin/adopt-mypy-and-stricter-linting  (`#208
  <https://github.com/AllenCellModeling/aicsimageio/pull/208>`_)
  [Jackson Maxfield Brown]
- admin/move-test-resources-hash-to-text-file  (`#202
  <https://github.com/AllenCellModeling/aicsimageio/pull/202>`_)
  [Jackson Maxfield Brown]
- admin/temp-bump-aicspylibczi  (`#194
  <https://github.com/AllenCellModeling/aicsimageio/pull/194>`_)
  [Jackson Maxfield Brown]
- Remove macOS11 and fix prepare-release [JacksonMaxfield]
- Attach dev0 to all references of version number [JacksonMaxfield]
- Configure dev_release bumpversion [JacksonMaxfield]
- Turn on doc building [JacksonMaxfield]
- Replace all references of master w/ main [JacksonMaxfield]
- Do not push docs on main yet [JacksonMaxfield]
- Second run of cookiecutter to ensure [JacksonMaxfield]
- Updates from cookiecutter [JacksonMaxfield]
- Force remove extra files [JacksonMaxfield]
- admin/fix-deadlink-in-documentation [JacksonMaxfield]
- admin/developer-documentation  (`#150
  <https://github.com/AllenCellModeling/aicsimageio/pull/150>`_)
  [Jackson Maxfield Brown, Josh Moore]
- Upgrade dependency versions [JacksonMaxfield]
- Working for small files [JacksonMaxfield]
- admin/proposed-api  (`#145
  <https://github.com/AllenCellModeling/aicsimageio/pull/145>`_)
  [Jackson Maxfield Brown]
- Remove everything [JacksonMaxfield]
- Remove dims setting entirely [JacksonMaxfield]
- Read dims, shape, and dtype from Tiff metadata in single shot
  [JacksonMaxfield]
- Change all self.dask_data.shape calls to self.shape in reader
  subclasses [JacksonMaxfield]
- Linting and formatting [JacksonMaxfield]
- Readers API stabilization, verbs -> functions, nouns -> properties
  [JacksonMaxfield]
- admin/update-from-cookiecutter  (`#136
  <https://github.com/AllenCellModeling/aicsimageio/pull/136>`_)
  [JacksonMaxfield]


v3.3.5 (2021-01-25)
-------------------

Fix
~~~
- bugfix/pin-readlif  (`#187
  <https://github.com/AllenCellModeling/aicsimageio/pull/187>`_)
  [Jackson Maxfield Brown]


v3.3.4 (2021-01-13)
-------------------
- ome-xml as string to ome tiff writer  (`#180
  <https://github.com/AllenCellModeling/aicsimageio/pull/180>`_)
  [toloudis]


v3.3.3 (2020-12-14)
-------------------

Fix
~~~
- bugfix/replace-napari-is-pyramid-with-multiscale  (`#172
  <https://github.com/AllenCellModeling/aicsimageio/pull/172>`_)
  [Dimitri Hürlimann, dimi-huer]

Other
~~~~~
- admin/support-py39  (`#169
  <https://github.com/AllenCellModeling/aicsimageio/pull/169>`_)
  [Jackson Maxfield Brown]


v3.3.2 (2020-11-17)
-------------------
- admin/update-base-reader-dep-versions  (`#156
  <https://github.com/AllenCellModeling/aicsimageio/pull/156>`_) [Jamie
  Sherman]


v3.3.1 (2020-09-23)
-------------------

Fix
~~~
- bugfix/tiff-rgb  (`#153
  <https://github.com/AllenCellModeling/aicsimageio/pull/153>`_) [Jamie
  Sherman]

Other
~~~~~
- admin/cleanup-readme  (`#149
  <https://github.com/AllenCellModeling/aicsimageio/pull/149>`_)
  [Jackson Maxfield Brown]


v3.3.0 (2020-09-09)
-------------------

New
~~~
- feature/use-in-memory-data-for-non-dask-calls  (`#148
  <https://github.com/AllenCellModeling/aicsimageio/pull/148>`_)
  [Jackson Maxfield Brown]


v3.2.3 (2020-06-23)
-------------------

New
~~~
- feature/reader-additions  (`#126
  <https://github.com/AllenCellModeling/aicsimageio/pull/126>`_)
  [JacksonMaxfield]


v3.2.2 (2020-06-11)
-------------------

New
~~~
- feature/enable-disable-dask  (`#124
  <https://github.com/AllenCellModeling/aicsimageio/pull/124>`_)
  [JacksonMaxfield]

Other
~~~~~
- admin/update-build-tooling  (`#123
  <https://github.com/AllenCellModeling/aicsimageio/pull/123>`_)
  [JacksonMaxfield]
- admin/switch-log-warning-to-warnings-warn  (`#122
  <https://github.com/AllenCellModeling/aicsimageio/pull/122>`_)
  [JacksonMaxfield]


v3.2.1 (2020-05-26)
-------------------

Fix
~~~
- bugfix/add-imagecodecs-dep  (`#120
  <https://github.com/AllenCellModeling/aicsimageio/pull/120>`_)
  [JacksonMaxfield]


v3.2.0 (2020-05-13)
-------------------

New
~~~
- feature/optimize-readers  (`#113
  <https://github.com/AllenCellModeling/aicsimageio/pull/113>`_)
  [JacksonMaxfield]
- feature/allow-sequence-in-get-data  (`#109
  <https://github.com/AllenCellModeling/aicsimageio/pull/109>`_)
  [JacksonMaxfield]
- feature/read-leica-lif-files  (`#99
  <https://github.com/AllenCellModeling/aicsimageio/pull/99>`_) [Jamie
  Sherman]

Fix
~~~
- bugfix/update-ome-spec  (`#116
  <https://github.com/AllenCellModeling/aicsimageio/pull/116>`_)
  [JacksonMaxfield]
- bugfix/set-sphinx-dep-upper-bound  (`#95
  <https://github.com/AllenCellModeling/aicsimageio/pull/95>`_)
  [JacksonMaxfield]

Other
~~~~~
- admin/benchmarks  (`#112
  <https://github.com/AllenCellModeling/aicsimageio/pull/112>`_)
  [JacksonMaxfield]
- admin/use-black-formatting  (`#108
  <https://github.com/AllenCellModeling/aicsimageio/pull/108>`_)
  [JacksonMaxfield]
- Update PR Template [Madison Bowden]
- admin/move-test-resources-to-s3  (`#94
  <https://github.com/AllenCellModeling/aicsimageio/pull/94>`_)
  [JacksonMaxfield]


v3.1.4 (2020-03-21)
-------------------

New
~~~
- feature/add-get-channel-names-to-base-reader  (`#88
  <https://github.com/AllenCellModeling/aicsimageio/pull/88>`_)
  [JacksonMaxfield]

Fix
~~~
- bugfix/reader-context-manager-top-level-import-error  (`#85
  <https://github.com/AllenCellModeling/aicsimageio/pull/85>`_)
  [JacksonMaxfield]


v3.1.3 (2020-03-11)
-------------------

Fix
~~~
- bugfix/delay-import-of-distributed-module  (`#83
  <https://github.com/AllenCellModeling/aicsimageio/pull/83>`_)
  [JacksonMaxfield]

Other
~~~~~
- admin/standardize-flake8-settings  (`#84
  <https://github.com/AllenCellModeling/aicsimageio/pull/84>`_)
  [JacksonMaxfield]


v3.1.2 (2020-03-06)
-------------------

New
~~~
- feature/get-physical-pixel-size  (`#80
  <https://github.com/AllenCellModeling/aicsimageio/pull/80>`_)
  [JacksonMaxfield]

Other
~~~~~
- admin/add-back-codecov  (`#81
  <https://github.com/AllenCellModeling/aicsimageio/pull/81>`_)
  [JacksonMaxfield]
- admin/changelog-link-to-prs  (`#77
  <https://github.com/AllenCellModeling/aicsimageio/pull/77>`_)
  [JacksonMaxfield]


v3.1.1 (2020-02-21)
-------------------

Fix
~~~
- bugfix/make-aicsimage-serializable (`#74
  <https://github.com/AllenCellModeling/aicsimageio/pull/74>`_)
  [JacksonMaxfield]
- bugfix/return-none-cluster  (`#73
  <https://github.com/AllenCellModeling/aicsimageio/pull/73>`_) [Jamie
  Sherman]

Other
~~~~~
- admin/auto-changelog  (`#75
  <https://github.com/AllenCellModeling/aicsimageio/pull/75>`_)
  [JacksonMaxfield]
- admin/test-py38  (`#76
  <https://github.com/AllenCellModeling/aicsimageio/pull/76>`_)
  [JacksonMaxfield]


v3.1.0 (2020-02-03)
-------------------

New
~~~
- feature/use-dask  (`#63
  <https://github.com/AllenCellModeling/aicsimageio/pull/63>`_)
  [JacksonMaxfield]

Fix
~~~
- Fix pypi publish action [Jackson Brown]
- bugfix/auto-doc-gen  (`#70
  <https://github.com/AllenCellModeling/aicsimageio/pull/70>`_)
  [JacksonMaxfield]


v3.0.7 (2019-11-05)
-------------------
- Remove make clean command from make docs call  (`#49
  <https://github.com/AllenCellModeling/aicsimageio/pull/49>`_)
  [JacksonMaxfield]
- populate_tiffdata should respect dimension order  (`#48
  <https://github.com/AllenCellModeling/aicsimageio/pull/48>`_)
  [toloudis]


v3.0.6 (2019-10-31)
-------------------

New
~~~
- Feature/physical pixel size  (`#43
  <https://github.com/AllenCellModeling/aicsimageio/pull/43>`_)
  [toloudis]

Fix
~~~
- fix imread bug and allow AICSImage class to close its reader  (`#44
  <https://github.com/AllenCellModeling/aicsimageio/pull/44>`_)
  [toloudis]


v3.0.5 (2019-10-30)
-------------------
- clean up from PR comments [Daniel Toloudis]
- add get_channel_names to AICSImage class [Daniel Toloudis]


v3.0.4 (2019-10-28)
-------------------
- add size getters to the AICSImage class  (`#38
  <https://github.com/AllenCellModeling/aicsimageio/pull/38>`_)
  [toloudis]


v3.0.3 (2019-10-25)
-------------------

Fix
~~~
- fix linting [Daniel Toloudis]
- fix png writer and tests [Daniel Toloudis]
- fix linter [Daniel Toloudis]
- fix png writer and tests [Daniel Toloudis]

Other
~~~~~
- Remove patch coverage check  (`#36
  <https://github.com/AllenCellModeling/aicsimageio/pull/36>`_)
  [JacksonMaxfield]
- pull request code review revisions [Dan Toloudis]
- Revert "fix png writer and tests" [Daniel Toloudis]
- use old default for dimension_order so that existing code does not
  break [Daniel Toloudis]
- add a unit test for dimension_order and refactor test_ome_tiff_writer
  [Daniel Toloudis]
- allow dimension order in ome-tiff writer [Daniel Toloudis]
- remove accidentally added file [Daniel Toloudis]
- Remove CRON from doc build workflow [Jackson Brown]
- Remove double builds from github actions [Jackson Brown]
- Wrap CRON string in quotes [Jackson Brown]
- Update CRON strings [Jackson Brown]
- Do not build documentation for tests module [Jackson Brown]
- Update makefile to remove all generated rst's on doc gen [Jackson
  Brown]
- Update czireader import so that it doesn't fail on etree [Jackson
  Brown]
- Move documentation badge to before codecov [JacksonMaxfield]
- Update readme to have doc badge [JacksonMaxfield]
- Add doc generation workflow [JacksonMaxfield]
- Add required documentation files and update requirements
  [JacksonMaxfield]
- Update task version pins to point at master / latest [JacksonMaxfield]


v3.0.2 (2019-10-11)
-------------------
- Pull in feedback from team [Jackson Brown]
- Update README to include known_dim functionality [Jackson Brown]
- Add test for invalid dim names [Jackson Brown]
- Resolves [gh-22], allow passing of known dim order to AICSImage
  [Jackson Brown]
- Resolves [gh-23], use OME-Tiff metadata to parse dim sizes and order
  [Jackson Brown]


v3.0.1 (2019-10-04)
-------------------
- Label Quickstart code block as python [Jackson Brown]
- Update setup.cfg to properly bumpversion [Jackson Brown]
- Rename build workflow to build master [Jackson Brown]
- Update to new cookiecutter gh templates after matts feedback [Jackson
  Brown]
- Remove cov report html from actions as not needed on remote [Jackson
  Brown]
- Add PR to test and lint action triggers [Jackson Brown]
- Remove references to quilt3distribute that were copied over [Jackson
  Brown]
- Update CI/CD, README badge, local developement, and contributing docs
  [Jackson Brown]
- CODE_OF_CONDUCT.md [Jamie Sherman]
