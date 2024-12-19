#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dictionary setting the "avalanche climate index". For now:
    Tromsoe & Soer-Troms:     maritime (index = 1)
    Lyngen:                   transitional (index = 2)
    Nord-Troms & Indre Troms: continental (index = 3)
"""

ava_clim_reg = {"NordTroms":3, "Lyngen":2, "Tromsoe":1, "SoerTroms":1, "IndreTroms":3}
ava_clim_code = {3009:3, 3010:2, 3011:1, 3012:1, 3013:3}