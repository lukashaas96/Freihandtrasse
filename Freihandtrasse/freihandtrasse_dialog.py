# -*- coding: utf-8 -*-
import os

from qgis.PyQt import uic
from qgis.PyQt import QtWidgets

# .ui-Datei laden, damit PyQt das Plugin mit den Elementen aus dem QTCreator füllen kann
FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'freihandtrasse_dialog.ui'))


class Freihandtrasse_Dialog(QtWidgets.QDialog, FORM_CLASS):
    def __init__(self, parent=None):
        """Constructor."""
        super(Freihandtrasse_Dialog, self).__init__(parent)
        self.setupUi(self)