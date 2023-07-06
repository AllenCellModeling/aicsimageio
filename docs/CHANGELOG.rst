Changelog
=========

v4.11.0 (2023-05-05)
--------------------

Fix
~~~
- Bugfix/slow tile retrieval  (`#486
  <https://github.com/AllenCellModeling/aicsimageio/pull/486>`_) [Sean
  LeRoy]
- Bugfix/mosaic lif merge error  (`#480
  <https://github.com/AllenCellModeling/aicsimageio/pull/480>`_) [Sean
  LeRoy]

Other
~~~~~
- Adjusting tiffile versioning  (`#487
  <https://github.com/AllenCellModeling/aicsimageio/pull/487>`_)
  [BrianWhitneyAI]
- store frame metadata when reading a specific scene  (`#492
  <https://github.com/AllenCellModeling/aicsimageio/pull/492>`_) [Joshua
  Gould]
- admin/support-py311  (`#446
  <https://github.com/AllenCellModeling/aicsimageio/pull/446>`_) [Brian
  Whitney, Eva Maxfield Brown, dmt, dmt, toloudis, toloudis]
- Extract S dim from OME data  (`#483
  <https://github.com/AllenCellModeling/aicsimageio/pull/483>`_)
  [Madison Swain-Bowden <bowdenm@spu.edu>    * Comment on possible
  dimension mismatch    ---------    Co-authored-by: Madison Swain-
  Bowden <bowdenm@spu.edu>, Sean LeRoy]


v4.10.0 (2023-04-11)
--------------------

New
~~~
- feature/github-action-stale  (`#474
  <https://github.com/AllenCellModeling/aicsimageio/pull/474>`_)
  [BrianWhitneyAI]
- feature/add-physical-pixel-size-to-tiff-reader  (`#456
  <https://github.com/AllenCellModeling/aicsimageio/pull/456>`_) [Talley
  Lambert]
- feature/zarrwriter  (`#381
  <https://github.com/AllenCellModeling/aicsimageio/pull/381>`_)
  [toloudis]

Fix
~~~
- Bugfix/empty scene name  (`#477
  <https://github.com/AllenCellModeling/aicsimageio/pull/477>`_)
  [BrianWhitneyAI, Sean LeRoy, Sean LeRoy
  <41307451+SeanLeRoy@users.noreply.github.com>    * Update
  aicsimageio/readers/czi_reader.py    Co-authored-by: Sean LeRoy
  <41307451+SeanLeRoy@users.noreply.github.com>    * more discriptive
  naming    ---------    Co-authored-by: Lukas Fan (----)
  <Lukas.Fan@imec.be>, lukas]
- bugfix: removing 2.11.0 threshold  (`#470
  <https://github.com/AllenCellModeling/aicsimageio/pull/470>`_)
  [BrianWhitneyAI]
- Bugfix: unclear warning for missing 'bfio' install.  (`#471
  <https://github.com/AllenCellModeling/aicsimageio/pull/471>`_)
  [BrianWhitneyAI]
- Bugfix: Upgrade minimum fsspec version  (`#473
  <https://github.com/AllenCellModeling/aicsimageio/pull/473>`_) [Sean
  LeRoy]
- Bugfix: Restrict tifffile version for now  (`#472
  <https://github.com/AllenCellModeling/aicsimageio/pull/472>`_) [Sean
  LeRoy]
- Bugfix: Fix tests not asserting error expected  (`#464
  <https://github.com/AllenCellModeling/aicsimageio/pull/464>`_) [Sean
  LeRoy]
- Bugfix/Docs-README-Mention-mvn/maven-requirement-for-bioformats_jar
  (`#463 <https://github.com/AllenCellModeling/aicsimageio/pull/463>`_)
  [Peter Sobolewski]
- Fix/build fix ometypes validation  (`#461
  <https://github.com/AllenCellModeling/aicsimageio/pull/461>`_)
  [toloudis]
- fix/fix_future_warnings  (`#453
  <https://github.com/AllenCellModeling/aicsimageio/pull/453>`_)
  [JasonYu1]

Other
~~~~~
- Restrict xarray version to support py3.8  (`#481
  <https://github.com/AllenCellModeling/aicsimageio/pull/481>`_) [Sean
  LeRoy]
- try to use optimized codepath through ome-types  (`#478
  <https://github.com/AllenCellModeling/aicsimageio/pull/478>`_)
  [toloudis]
- Quality of Life: Make pip install platform independent  (`#466
  <https://github.com/AllenCellModeling/aicsimageio/pull/466>`_) [Sean
  LeRoy]
- More descriptive scene index error message.  (`#458
  <https://github.com/AllenCellModeling/aicsimageio/pull/458>`_) [Philip
  Garrison]


v4.9.4 (2022-12-06)
-------------------

Fix
~~~
- bugfix/Fix `UnboundLocalError` in TiffGlobReader  (`#449
  <https://github.com/AllenCellModeling/aicsimageio/pull/449>`_) [Ian
  Hunt-Isaak, Madison Swain-Bowden <bowdenm@spu.edu>    * get typing
  correct    * apply black + fix test typing    Co-authored-by: Madison
  Swain-Bowden <bowdenm@spu.edu>]


v4.9.3 (2022-11-15)
-------------------

Fix
~~~
- bugfix/handle-tiff-i-dim  (`#445
  <https://github.com/AllenCellModeling/aicsimageio/pull/445>`_)
  [toloudis]
- bugfix/single-dimension-index-requested-in-transform-return  (`#438
  <https://github.com/AllenCellModeling/aicsimageio/pull/438>`_)
  [toloudis]
- fix/position-names-in-nd2-tests  (`#437
  <https://github.com/AllenCellModeling/aicsimageio/pull/437>`_) [Talley
  Lambert]

Other
~~~~~
- admin/citation-update  (`#440
  <https://github.com/AllenCellModeling/aicsimageio/pull/440>`_) [Eva
  Maxfield Brown]


v4.9.2 (2022-08-24)
-------------------

New
~~~
- feature/add-physical-pixel-sizes-param-to-array-like  (`#426
  <https://github.com/AllenCellModeling/aicsimageio/pull/426>`_)
  [Guilherme Pires, Guilherme Pires]

Fix
~~~
- bugfix/convert-dimension-spec-lists-to-slices-when-possible  (`#429
  <https://github.com/AllenCellModeling/aicsimageio/pull/429>`_)
  [toloudis]
- bugfix/czi-scene-selection-for-inconsistent-scenes-regression  (`#432
  <https://github.com/AllenCellModeling/aicsimageio/pull/432>`_)
  [toloudis]

Other
~~~~~
- Remove CZI extra install that doesn't come with CZI [Eva Maxfield
  Brown]
- admin/include-fsspec-dep-for-czi-in-readme  (`#433
  <https://github.com/AllenCellModeling/aicsimageio/pull/433>`_) [Eva
  Maxfield Brown]


v4.9.1 (2022-08-02)
-------------------

Fix
~~~
- bugfix/fsspec-local-file-opener-cpp-buffer-for-czi  (`#425
  <https://github.com/AllenCellModeling/aicsimageio/pull/425>`_) [Eva
  Maxfield Brown]
- Fix/czireader float32 dtype  (`#423
  <https://github.com/AllenCellModeling/aicsimageio/pull/423>`_)
  [toloudis]
- bugfix/extract-czi-channel-names-more-safely  (`#418
  <https://github.com/AllenCellModeling/aicsimageio/pull/418>`_)
  [toloudis]
- bugfix/czi-scene-indexing  (`#417
  <https://github.com/AllenCellModeling/aicsimageio/pull/417>`_)
  [toloudis]
- bugfix/dynamic-dimension-typing  (`#419
  <https://github.com/AllenCellModeling/aicsimageio/pull/419>`_) [Eva
  Maxfield Brown]

Other
~~~~~
- admin/ignore-fsspec-2022.7.0  (`#424
  <https://github.com/AllenCellModeling/aicsimageio/pull/424>`_) [Eva
  Maxfield Brown]
- admin/add-fsspec-to-upstream-checks  (`#422
  <https://github.com/AllenCellModeling/aicsimageio/pull/422>`_) [Eva
  Maxfield Brown]


v4.9.0 (2022-07-19)
-------------------

New
~~~
- feature/image-container  (`#415
  <https://github.com/AllenCellModeling/aicsimageio/pull/415>`_) [Eva
  Maxfield Brown]
- feature/lower-log-level-of-OME-TIFF-read-errors  (`#414
  <https://github.com/AllenCellModeling/aicsimageio/pull/414>`_) [Eva
  Maxfield Brown]
- feature/get-stack  (`#403
  <https://github.com/AllenCellModeling/aicsimageio/pull/403>`_) [John
  Russell]

Other
~~~~~
- Update names, emails, and meta [evamaxfield]


v4.8.0 (2022-05-26)
-------------------

New
~~~
- feature-and-admin/create-aicsimage-objects-with-fs-kwargs-and-remove-
  need-for-creds  (`#407
  <https://github.com/AllenCellModeling/aicsimageio/pull/407>`_)
  [Jackson Maxfield Brown]

Fix
~~~
- bugfix/pass-series-index-in-biofile-init  (`#401
  <https://github.com/AllenCellModeling/aicsimageio/pull/401>`_)
  [Jackson Maxfield Brown]

Other
~~~~~
- admin/prepare-bioformats_reader-to-work-with-new-bioformats_jar-based-
  on-scyjava  (`#402
  <https://github.com/AllenCellModeling/aicsimageio/pull/402>`_) [Talley
  Lambert]
- admin/add-test-upstreams-action  (`#406
  <https://github.com/AllenCellModeling/aicsimageio/pull/406>`_)
  [Jackson Maxfield Brown, Matthew Rocklin]
- admin/ship-mypy-type-annotations-drop-py37-add-py310-manage-test-
  dependencies-separate-for-each-tox-env  (`#397
  <https://github.com/AllenCellModeling/aicsimageio/pull/397>`_)
  [Jackson Maxfield Brown]


v4.7.0 (2022-04-19)
-------------------

New
~~~
- feature/tiledtiffreader  (`#387
  <https://github.com/AllenCellModeling/aicsimageio/pull/387>`_)
  [Nicholas-Schaub]

Other
~~~~~
- Upgrade black version [JacksonMaxfield]


v4.6.4 (2022-03-18)
-------------------

Fix
~~~
- bugfix/update-czi-to-ome-xslt-for-channel-id  (`#389
  <https://github.com/AllenCellModeling/aicsimageio/pull/389>`_)
  [Griffin Fujioka, Griffin Fujioka]


v4.6.3 (2022-03-03)
-------------------

Fix
~~~
- bugfix/add-logic-to-ensure-OME-XML-plane-elements-occur-last  (`#385
  <https://github.com/AllenCellModeling/aicsimageio/pull/385>`_)
  [Nicholas-Schaub]


v4.6.2 (2022-03-01)
-------------------
- admin/update-czi-to-ome-xslt-submodule  (`#382
  <https://github.com/AllenCellModeling/aicsimageio/pull/382>`_)
  [Jackson Maxfield Brown]


v4.6.1 (2022-03-01)
-------------------

Fix
~~~
- bugfix/czi-physical-size  (`#384
  <https://github.com/AllenCellModeling/aicsimageio/pull/384>`_)
  [emay2022]


v4.6.0 (2022-02-22)
-------------------

Fix
~~~
- bugfix/more-info-and-help-on-corrupt-file  (`#380
  <https://github.com/AllenCellModeling/aicsimageio/pull/380>`_)
  [Jackson Maxfield Brown]
- bugfix/reader-selection-fixes  (`#367
  <https://github.com/AllenCellModeling/aicsimageio/pull/367>`_)
  [Jackson Maxfield Brown]

Other
~~~~~
- admin/remove-czi-install-pattern  (`#376
  <https://github.com/AllenCellModeling/aicsimageio/pull/376>`_)
  [Jackson Maxfield Brown]
- admin/bump-nd2-v0.2.0  (`#379
  <https://github.com/AllenCellModeling/aicsimageio/pull/379>`_) [Talley
  Lambert]


v4.5.2 (2021-12-16)
-------------------
- Update XSLT subbmodule  (`#365
  <https://github.com/AllenCellModeling/aicsimageio/pull/365>`_) [Matte
  Bailey]
- Add instructions for updating the submodule  (`#364
  <https://github.com/AllenCellModeling/aicsimageio/pull/364>`_) [Matte
  Bailey]


v4.5.1 (2021-12-08)
-------------------

New
~~~
- feature/czi-subblock-metadata  (`#353
  <https://github.com/AllenCellModeling/aicsimageio/pull/353>`_) [Matte
  Bailey]

Fix
~~~
- bugfix/pin-nd2-dep  (`#358
  <https://github.com/AllenCellModeling/aicsimageio/pull/358>`_) [Talley
  Lambert]
- fix/only-close-biofiofile-on-exit-if-not-open-on-entry  (`#341
  <https://github.com/AllenCellModeling/aicsimageio/pull/341>`_) [Talley
  Lambert]

Other
~~~~~
- Update submodule commit to latest  (`#363
  <https://github.com/AllenCellModeling/aicsimageio/pull/363>`_) [Matte
  Bailey]
- Add missing char to workflow file  (`#362
  <https://github.com/AllenCellModeling/aicsimageio/pull/362>`_) [Matte
  Bailey]
- Add ability to manually trigger build of main  (`#359
  <https://github.com/AllenCellModeling/aicsimageio/pull/359>`_) [Matte
  Bailey]
- allow metadata extraction from czi files with no subblocks  (`#360
  <https://github.com/AllenCellModeling/aicsimageio/pull/360>`_)
  [toloudis]


v4.5.0 (2021-11-04)
-------------------

New
~~~
- feature/add-stored-mm-indexer-function-to-glob-reader  (`#346
  <https://github.com/AllenCellModeling/aicsimageio/pull/346>`_) [Ian
  Hunt-Isaak]
- feature/add-getitem-to-dimensions-object  (`#347
  <https://github.com/AllenCellModeling/aicsimageio/pull/347>`_) [Ian
  Hunt-Isaak]
- feature/allow-bioformats-x-y-chunking   (`#336
  <https://github.com/AllenCellModeling/aicsimageio/pull/336>`_) [Heath
  Patterson]
- feature/glob-reader  (`#303
  <https://github.com/AllenCellModeling/aicsimageio/pull/303>`_) [John
  Russell, Madison Swain-Bowden]

Other
~~~~~
- admin/update-pr-template-remove-link-to-gh-12  (`#342
  <https://github.com/AllenCellModeling/aicsimageio/pull/342>`_)
  [Madison Swain-Bowden]


v4.4.0 (2021-10-12)
-------------------

New
~~~
- feature/add-native-deltavision-reader  (`#333
  <https://github.com/AllenCellModeling/aicsimageio/pull/333>`_) [Talley
  Lambert]
- feature/add-native-nd2-reader  (`#317
  <https://github.com/AllenCellModeling/aicsimageio/pull/317>`_) [Talley
  Lambert]


v4.3.0 (2021-10-08)
-------------------

New
~~~
- feature/import-any-reader-from-readers  (`#326
  <https://github.com/AllenCellModeling/aicsimageio/pull/326>`_) [Talley
  Lambert]

Other
~~~~~
- admin/remove-lif-from-standard-install  (`#332
  <https://github.com/AllenCellModeling/aicsimageio/pull/332>`_)
  [Jackson Maxfield Brown]
- admin/remove-bioformats-extra  (`#329
  <https://github.com/AllenCellModeling/aicsimageio/pull/329>`_) [Talley
  Lambert]
- admin/bump-readlif-for-multi-scene  (`#327
  <https://github.com/AllenCellModeling/aicsimageio/pull/327>`_)
  [psobolewskiPhD]
- docs/more-detailed-bioformats-install-instruction-on-error  (`#324
  <https://github.com/AllenCellModeling/aicsimageio/pull/324>`_) [Talley
  Lambert]


v4.2.0 (2021-09-27)
-------------------

New
~~~
- feature/bioformats-reader  (`#306
  <https://github.com/AllenCellModeling/aicsimageio/pull/306>`_) [Talley
  Lambert]
- feature/metadata-module  (`#292
  <https://github.com/AllenCellModeling/aicsimageio/pull/292>`_) [Matte
  Bailey]

Fix
~~~
- bugfix/resolve-conflict-between-format-impls-and-determine-reader
  (`#318 <https://github.com/AllenCellModeling/aicsimageio/pull/318>`_)
  [Jackson Maxfield Brown]
- bugfix/catch-channel-colors-errors-from-typing  (`#299
  <https://github.com/AllenCellModeling/aicsimageio/pull/299>`_)
  [Jackson Maxfield Brown]
- bugfix/asv-with-submodules  (`#297
  <https://github.com/AllenCellModeling/aicsimageio/pull/297>`_)
  [Jackson Maxfield Brown]
- bugfix/pass-aws-creds-to-tox  (`#296
  <https://github.com/AllenCellModeling/aicsimageio/pull/296>`_) [Matte
  Bailey]
- bugfix/checkout-submodules-for-doc-building  (`#295
  <https://github.com/AllenCellModeling/aicsimageio/pull/295>`_)
  [Jackson Maxfield Brown]

Other
~~~~~
- docs/update-biofile-docstring-add-bf-options-param  (`#322
  <https://github.com/AllenCellModeling/aicsimageio/pull/322>`_) [Talley
  Lambert]
- admin/exclude-benchmarks-from-package-install  (`#319
  <https://github.com/AllenCellModeling/aicsimageio/pull/319>`_)
  [Jackson Maxfield Brown]
- admin/loosen-version-pins  (`#320
  <https://github.com/AllenCellModeling/aicsimageio/pull/320>`_)
  [Jackson Maxfield Brown]
- admin/add-test-resources-caching  (`#313
  <https://github.com/AllenCellModeling/aicsimageio/pull/313>`_)
  [Jackson Maxfield Brown]
- admin/add-tox-work-dir-env-var-opt-to-tox-ini  (`#310
  <https://github.com/AllenCellModeling/aicsimageio/pull/310>`_) [Talley
  Lambert]
- Add .pre-commit-config.yaml  (`#308
  <https://github.com/AllenCellModeling/aicsimageio/pull/308>`_) [Talley
  Lambert]
- admin/add-custom-reader-addition-docs  (`#305
  <https://github.com/AllenCellModeling/aicsimageio/pull/305>`_)
  [Jackson Maxfield Brown, Madison Swain-Bowden <bowdenm@spu.edu>    *
  Grammer on object's    Co-authored-by: Madison Swain-Bowden
  <bowdenm@spu.edu>    * Better CziReader custom dependency link    Co-
  authored-by: Madison Swain-Bowden <bowdenm@spu.edu>    * Remove extra
  words    Co-authored-by: Madison Swain-Bowden <bowdenm@spu.edu>    *
  Better wording on benchmark additions    Co-authored-by: Madison
  Swain-Bowden <bowdenm@spu.edu>    * Clean up Reader class impl section
  Co-authored-by: Madison Swain-Bowden <bowdenm@spu.edu>]


v4.1.0 (2021-08-10)
-------------------

New
~~~
- feature/ome-metadata-xslt-spec  (`#289
  <https://github.com/AllenCellModeling/aicsimageio/pull/289>`_)
  [Jackson Maxfield Brown]

Fix
~~~
- bugfix/use-dict-of-tiff-tags  (`#293
  <https://github.com/AllenCellModeling/aicsimageio/pull/293>`_)
  [Jackson Maxfield Brown]
- bugfix/invert-lif-scales  (`#288
  <https://github.com/AllenCellModeling/aicsimageio/pull/288>`_)
  [Jackson Maxfield Brown]

Other
~~~~~
- admin/loosen-any-non-zero-semver-deps  (`#294
  <https://github.com/AllenCellModeling/aicsimageio/pull/294>`_)
  [Jackson Maxfield Brown]


v4.0.5 (2021-07-15)
-------------------

Fix
~~~
- bugfix/ome-tiff-ome-not-set  (`#284
  <https://github.com/AllenCellModeling/aicsimageio/pull/284>`_)
  [Jackson Maxfield Brown]

Other
~~~~~
- admin/deprecate-chunk-by-dims  (`#286
  <https://github.com/AllenCellModeling/aicsimageio/pull/286>`_)
  [Jackson Maxfield Brown]
- admin/bump-min-tifffile  (`#285
  <https://github.com/AllenCellModeling/aicsimageio/pull/285>`_)
  [Jackson Maxfield Brown]


v4.0.4 (2021-07-12)
-------------------

New
~~~
- feature/allow-set-scene-by-index  (`#272
  <https://github.com/AllenCellModeling/aicsimageio/pull/272>`_)
  [Jackson Maxfield Brown]

Other
~~~~~
- admin/finalizing-contributor-pr-process  (`#276
  <https://github.com/AllenCellModeling/aicsimageio/pull/276>`_)
  [Jackson Maxfield Brown]
- admin/local-tests-by-default  (`#273
  <https://github.com/AllenCellModeling/aicsimageio/pull/273>`_)
  [Jackson Maxfield Brown]


v4.0.3 (2021-07-05)
-------------------

Fix
~~~
- bugfix/missing-tiff-description-tag  (`#271
  <https://github.com/AllenCellModeling/aicsimageio/pull/271>`_)
  [Jackson Maxfield Brown]
- bugfix/channel-names-array-expansion  (`#265
  <https://github.com/AllenCellModeling/aicsimageio/pull/265>`_)
  [toloudis]

Other
~~~~~
- @rcasero-feature/add-python-3.7-support  (`#270
  <https://github.com/AllenCellModeling/aicsimageio/pull/270>`_)
  [Jackson Maxfield Brown, Ramón Casero]


v4.0.2 (2021-06-22)
-------------------

Fix
~~~
- bugfix/mosaic-tile-reconstruction-for-multi-scene-mosaics  (`#260
  <https://github.com/AllenCellModeling/aicsimageio/pull/260>`_)
  [toloudis]

Other
~~~~~
- admin/bump-aicspylibczi-dep-version  (`#261
  <https://github.com/AllenCellModeling/aicsimageio/pull/261>`_)
  [Jackson Maxfield Brown]
- admin/add-doi  (`#258
  <https://github.com/AllenCellModeling/aicsimageio/pull/258>`_)
  [Jackson Maxfield Brown]
- docs/add-missing-czi-reading-to-mosaic-support  (`#256
  <https://github.com/AllenCellModeling/aicsimageio/pull/256>`_)
  [Jackson Maxfield Brown]
- docs/fix-physical-pixel-sizes-typo  (`#253
  <https://github.com/AllenCellModeling/aicsimageio/pull/253>`_)
  [Jackson Maxfield Brown]


v4.0.1 (2021-06-08)
-------------------

Fix
~~~
- bugfix/coords-and-floating-point-math-and-czi-scene-naming  (`#250
  <https://github.com/AllenCellModeling/aicsimageio/pull/250>`_)
  [Jackson Maxfield Brown]


v4.0.0 (2021-06-07)
-------------------

New
~~~
- feature/default-to-pixel-size-none  (`#246
  <https://github.com/AllenCellModeling/aicsimageio/pull/246>`_)
  [Jackson Maxfield Brown]
- feature/czi-reader  (`#231
  <https://github.com/AllenCellModeling/aicsimageio/pull/231>`_)
  [JacksonMaxfield, Jamie Sherman]
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
- bugfix/support-rgb-mosaic-czi  (`#247
  <https://github.com/AllenCellModeling/aicsimageio/pull/247>`_)
  [Jackson Maxfield Brown]
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
- admin/remove-dev-release-infra  (`#248
  <https://github.com/AllenCellModeling/aicsimageio/pull/248>`_)
  [Jackson Maxfield Brown]
- admin/4.0-release-prep-and-benchmark-upgrades  (`#244
  <https://github.com/AllenCellModeling/aicsimageio/pull/244>`_)
  [Jackson Maxfield Brown]
- admin/add-install-from-git-to-readme  (`#240
  <https://github.com/AllenCellModeling/aicsimageio/pull/240>`_) [Ramón
  Casero]
- admin/2021-dask-summit-presentation  (`#236
  <https://github.com/AllenCellModeling/aicsimageio/pull/236>`_)
  [Jackson Maxfield Brown]
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
