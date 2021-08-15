# -*- coding: utf-8 -*-
from .freihandtrasse import Freihandtrasse

def classFactory(iface):
	return Freihandtrasse(iface)

