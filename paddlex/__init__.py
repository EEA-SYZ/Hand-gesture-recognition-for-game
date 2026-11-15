# -*- coding: utf-8 -*-
"""PaddleX 1.x deploy API compatibility layer.

The original project imports ``from paddlex.deploy import Predictor``.  That
API belongs to PaddleX 1.3.x, which cannot be installed on Python 3.12 and
depends on the removed ``paddle.fluid`` module.  This package re-implements
the small slice of the API the game needs on top of the modern
``paddle.inference`` API, reading ``model.yml`` for labels and transforms.

It is intentionally placed in the project root (instead of being installed)
so ``python plane_cnn.py`` picks it up without any extra configuration.
"""

__version__ = "1.3.1-compat"
