Changelog
=========

All notable changes to `torchiva
<https://github.com/fakufaku/torchiva>`_ will be documented in this file.

The format is based on `Keep a
Changelog <http://keepachangelog.com/en/1.0.0/>`__ and this project
adheres to `Semantic Versioning <http://semver.org/spec/v2.0.0.html>`_.

`Unreleased`_
-------------

Nothing yet.

`0.1.1`_ - 2022-11-15
---------------------

Added
~~~~~

- Adds a built-in separation function in ``torchiva.separate``
- Adds a new ``torchiva.load_separator`` function that is specific to
  the model format used in ``examples/export_model.py`` and ``trained_models/tiss``

Removed
~~~~~~~

- Some unrelated functions for conversion between spherical and cartesian
  coordinates in ``torchiva.utils``


.. _Unreleased: https://github.com/LCAV/pyroomacoustics/compare/v0.1.1...master
.. _0.1.1: https://github.com/LCAV/pyroomacoustics/compare/v0.1.0...v0.1.1
