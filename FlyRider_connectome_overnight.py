# -*- coding: utf-8 -*-
"""Compatibility launcher; overnight execution is implemented in FlyRider_connectome.py."""
import sys

if "--overnight" not in sys.argv:
    sys.argv.append("--overnight")

from FlyRider_connectome import main

if __name__ == "__main__":
    main()
