# -*- coding: utf-8 -*-

# ||||||||||||||||||||||||||||||| Steuerparameter zum erstellen der Benutzeroberfläche |||||||||||||||||||||||||||||||||

gui_erstellen = True

# ||||||||||||||||||||||||||||||||||||||||||||||||||||| Importe ||||||||||||||||||||||||||||||||||||||||||||||||||||||||

# Benötigte QGIS- und Qt-Bibliotheken
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction
from PyQt5.QtCore import QVariant
from qgis.core import Qgis, QgsProject, QgsCoordinateTransform, QgsCoordinateReferenceSystem, QgsPointXY, QgsVectorLayer, QgsFeature, QgsGeometry, QgsField, QgsWkbTypes
# Icon
from .resources import *
# GUI
from .freihandtrasse_dialog import Freihandtrasse_Dialog
# NumPy, PyPlot und F- bzw. T-Verteilung importieren
import numpy as np
import matplotlib.pyplot as plt

# ||||||||||||||||||||||||||||||||||||||||||||||||| globale Methoden |||||||||||||||||||||||||||||||||||||||||||||||||||

def gleitender_mittelwert(array_in, glaettungsfaktor):
    """Berechnet einen gleitenden Mittelwert der zweiten Spalte des Arrays und weist die erste Spalte entsprechend zu.
       Der Randbereich wird gelöscht.
       array_in: Array mit Längen und Krümmungen, Längen werden übernommen, Krümmungen geglättet
       glaettungsfaktor: Gleitender Mittelwert über glaettungsfaktor-viele Punkte"""
    i = 0
    # Leeres Output-Array
    array_out = np.zeros(shape=(np.shape(array_in)[0] - glaettungsfaktor + 1, 2))
    while i < np.shape(array_out)[0]:
        # Länge zuordnen: mittlerer Punkt, bei geradem glaettungsfaktor nächsthöherer Punkt (vermeiden)
        array_out[i][0] = array_in[int(glaettungsfaktor / 2) + i][0]
        summe = 0
        j = 0
        # Kruemmungen /Richtungswinkel glaetten
        while j < glaettungsfaktor:
            summe += array_in[i + j][1]
            j += 1
        array_out[i][1] = summe / glaettungsfaktor
        i += 1
    return array_out

def strecke(rechtsa, rechtse, hocha, hoche):
    """Berechnet die Strecke zwischen zwei Punkten
       rechtsa: Rechtswert Standpunkt,
       rechtse: Rechtswert Zielpunkt
       hocha: Hochwert Standpunkt
       hoche: Hochwert Zielpunkt"""
    return np.sqrt(np.square(rechtse - rechtsa) + np.square(hoche - hocha))

def riwi(rechtsa, rechtse, hocha, hoche):
    """Berechnet den Richtungswinkel zwischen zwei Punkten [rad] (geodätisch)
       rechtsa: Rechtswert Standpunkt,
       rechtse: Rechtswert Zielpunkt
       hocha: Hochwert Standpunkt
       hoche: Hochwert Zielpunkt"""
    t = np.arctan2(rechtse - rechtsa, hoche - hocha)
    if t < 0:
        return t + 2 * np.pi
    elif t > 2 * np.pi:
        return t - 2 * np.pi
    else:
        return t

def layer2array(importLayer):
    """Layer mit Punktgeometrien in Array mit E- und N-Koordinaten in UTM32 speichern
       selectedLayer: Ausgewählter Layer mit digitalisierten Punkten in QGIS"""
    # leeres Array erstellen
    layerpunkte = np.zeros(shape=(importLayer.featureCount(), 2))
    # Transformation in UTM32 vorbereiten
    crs_quell = importLayer.crs()  # Quell-Layer-CRS
    crs_ziel = QgsCoordinateReferenceSystem.fromEpsgId(25832)  # Ziel-CRS: UTM32
    xform = QgsCoordinateTransform(crs_quell, crs_ziel, QgsProject.instance())
    # Alle Features des Layers
    features = importLayer.getFeatures()
    # layerpunkte füllen (Punkte in UTM)
    i = 0
    for feature in features:
        punkt = xform.transform(QgsPointXY(feature.geometry().asPoint().x(), feature.geometry().asPoint().y()))
        layerpunkte[i][0] = punkt.x()
        layerpunkte[i][1] = punkt.y()
        i = i + 1
    return layerpunkte

def punkte2layer(trassenpunkte):
    """Erstellt einen Qgis-Vektorlayer (ETRS89/UTM32) aus einer Liste mit Objekten der Klasse Trassenpunkt.
       Übernimmt die Attribute."""
    output_layer = QgsVectorLayer('Point?crs=epsg:25832', 'Trassenpunkte', 'memory')
    QgsProject.instance().addMapLayers([output_layer])
    output_layer.dataProvider().addAttributes([QgsField("Station", QVariant.String)])
    output_layer.dataProvider().addAttributes([QgsField("Punktart", QVariant.String)])
    output_layer.updateFields()
    felder = output_layer.fields()
    featureliste = []
    for pkt in trassenpunkte:
        feature = QgsFeature()
        feature.setFields(felder)
        punkt = QgsPointXY(pkt.x, pkt.y)
        feature.setGeometry(QgsGeometry.fromPointXY(punkt))
        feature['Station'] = pkt.pnr
        if pkt.schluessel == 0:
            feature['Punktart'] = "Zwischenpunkt"
        else:
            feature['Punktart'] = "Hauptpunkt"
        featureliste.append(feature)
    output_layer.dataProvider().addFeatures(featureliste)
    output_layer.commitChanges()
    return output_layer

def kruemmungberechnen1(glatt_k, layerpunkte):
    """Krümmungen über Pfeilhöhenpolygon berechnen
       glatt_k: Glättungsfaktor der Krümmungen (Berechnung der Pfeilhöhe aus glatt_k-vielen Punkten nach links und glatt_k-vielen Punkten nach rechts des beobachteten Punktes)
       pointsarray: Array mit digitalisierten Punkten in QGIS"""

    # leeres Array erstellen
    kruemmungen = np.zeros(shape=(np.shape(layerpunkte)[0] - 2 * glatt_k, 2))
    # Länge l = Addition der Sehnen bis zum Punkt vor dem ersten, zu dem die Krümmung berechnet wird
    i = 0
    l = 0
    while i < glatt_k - 1:
        l = l + strecke(layerpunkte[i][0], layerpunkte[i + 1][0], layerpunkte[i][1], layerpunkte[i + 1][1])
        i += 1
    i = glatt_k
    # Matrix mit Krümmungen und Längen
    for kruemmung in kruemmungen:
        s2 = strecke(layerpunkte[i, 0], layerpunkte[i + glatt_k, 0], layerpunkte[i, 1], layerpunkte[i + glatt_k, 1])
        t_s1 = riwi(layerpunkte[i - glatt_k, 0], layerpunkte[i, 0], layerpunkte[i - glatt_k, 1], layerpunkte[i, 1])
        t_s = riwi(layerpunkte[i - glatt_k, 0], layerpunkte[i + glatt_k, 0], layerpunkte[i - glatt_k, 1],
                   layerpunkte[i + glatt_k, 1])
        k = np.sin(t_s - t_s1) * 2 / s2 * 1000
        kruemmungen[i - glatt_k, 1] = k
        l = l + strecke(layerpunkte[i - 1, 0], layerpunkte[i, 0], layerpunkte[i - 1, 1], layerpunkte[i, 1])
        kruemmungen[i - glatt_k, 0] = l
        i = i + 1
    return kruemmungen

def kruemmungberechnen2(glatt_t, glatt_k, layerpunkte):
    """Krümmungen über Richtungswinkelpolygon berechnen
       glatt_t: Glättungsfaktor der Richtungswinkel
       glatt_k: Glättungsfaktor der Krümmungen
       pointsarray: Array mit digitalisierten Punkten in QGIS"""

    # leeres Array fuer Richtungswinkel erstellen
    riwiarray = np.zeros(shape=(np.shape(layerpunkte)[0] - 2, 2))

    # Ab Index 1 Array durch Schleife füllen
    l = 0.0
    i = 1
    while i <= np.shape(riwiarray)[0]:
        l = l + strecke(layerpunkte[i - 1][0], layerpunkte[i][0], layerpunkte[i - 1][1], layerpunkte[i][1])
        t = riwi(layerpunkte[i - 1][0], layerpunkte[i + 1][0], layerpunkte[i - 1][1], layerpunkte[i + 1][1])
        riwiarray[i-1][0] = l
        riwiarray[i-1][1] = t
        i += 1
    # Richtungswinkel glätten (gleitender Mittelwert über glatt_t Punkte)
    riwiglatt = gleitender_mittelwert(riwiarray, glatt_t)
    # leeres Array fuer Kruemmungen erstellen
    kruemmungen = np.zeros(shape=(np.shape(riwiglatt)[0] - 2, 2))
    # Krümmungen aus Richtungswinkeln (=> 1.Ableitung) errechnen
    i = 1
    while i <= np.shape(kruemmungen)[0]:
        kruemmungen[i-1][0] = riwiglatt[i][0]
        kruemmungen[i-1][1] = (riwiglatt[i + 1][1] - riwiglatt[i-1][1]) / (riwiglatt[i + 1][0] - riwiglatt[i-1][0]) * 1000
        i += 1
    # Krümmungen glätten (gleitender Mittelwert über glatt_k Punkte
    kruemmungenglatt = gleitender_mittelwert(kruemmungen, glatt_k)
    return kruemmungenglatt

def ausgleichung(X, y):
    """Berechnet die Matrizen einer Ausgleichung bei gegebener Modellmatrix X und gegebenem Beobachtungsvektor y.
       Ausgabe von:
            ausgeglichenen Unbekannten (cdach)
            Freiheitsgraden (f)
            Kovarianzmatrix der ausgeglichenen Unbekannten (Scdachcdach)
            Varianz der Gewichtseinheit a posteriori (s02post)
            Normalgleichungsmatrix (N)"""
    N = np.dot(X.transpose(), X)
    n = np.dot(X.transpose(), y)
    cdach = np.dot(np.linalg.inv(N), n)
    v = np.dot(X, cdach) - y
    Qcdachcdach = np.linalg.inv(N)
    f = np.shape(y)[0] - np.shape(cdach)[0]
    s02post = np.dot(np.transpose(v), v) / f
    Scdachcdach = s02post * Qcdachcdach
    return np.array((cdach, f, Scdachcdach, s02post, N))

def ausgleichende_geraden_2(kruemmungen, anzahl_pkte_parallelen, prozentsatz_be, anzahl_bereiche, kreis2gerade, abstand_neues_be, l_ges, min_l_ei):
    """Methode, die anhand eines geglätteten Krümmungsbildes einer eng und gleichmäßig digitalisierten Trasse mit Bogenelementen einer Größenordnung die Parameter der Bogenelemente ermittelt
        kruemmungen: Matrix mit geglätteten Krümmungen und zugehörigen Stationen [l, k]
        anzahl_pkte_parallelen: Anzahl der (~aufeinenderfolgenden) Punkte innerhalb eines K-Bereichs, ab der das Programm eine Achsparallele erkennt
        prozentsatz_be: Prozentsatz aller Punkte eines für ein Bogenelement ermittelten Abschnitts, der (symmetrisch) zur Berechnung der Koeffizienten genutzt wird [0;1]
        anzahl_bereiche: Anzahl der k-Bereiche, in die das Krümmungsbild zur Ermittlung der Achsparallelen aufgeteilt wird
        kreis2gerade: Prozentsatz des Medians der Krümmungen, ab wann eine Achsparallele mit Krümmung != 0 zur Achsparallele mit Krümmung = 0 wird [0;1]
        abstand_neues_be: Faktor, mit dem der Median der Punktabstände multipliziert wird, um zu testen, ob ein Punkt noch zur aktuellen Achsparallele gehört, oder nicht
        lges: Gesamtsummen der Sehnen zwischen ALLEN Punkten
        min_l_ei: Minimale Laenge eines Kreisbogens zwischen zwei gleichsinnig gekrümmten Klothoiden (Eilinien-Schranke)
        """

    # Aufsplitten des kruemmungen-Arrays in zwei Matrizen, die jeweils nur die Krümmungen und die Längen enthalten
    nurkruemmungen = np.array_split(kruemmungen, [1], axis=1)[1]
    nurlaengen = np.array_split(kruemmungen, [1], axis=1)[0]

    # Maximale und minimale Krümmung sowie Spannweite der maximalen und minimalen Krümmung
    max_k = nurkruemmungen.max()
    min_k = nurkruemmungen.min()
    wertebereich_k = max_k - min_k

    # Array mit Strecken von Punkt zu Punkt
    strecken = np.empty([1])
    i = 1
    while i < np.shape(kruemmungen)[0]:
        strecken = np.vstack((strecken, nurlaengen[i] - nurlaengen[i - 1]))
        i += 1

    # Median der Strecken von Punkt zu Punkt und der Beträge der Krümmungen
    median_l = np.median(strecken)
    median_k = np.median(abs(nurkruemmungen))

    # Array mit horizontalen Achsparallelen
    achsparallelen = np.empty([0, 2])
    i = 0
    while i < anzahl_bereiche:
        j = 0
        # Achsparallele = zusammenhängender Teil eines k-Bereichs
        bereich = np.empty([0, 2])
        while j < np.shape(kruemmungen)[0]:
            if i < anzahl_bereiche - 1:
                # Klassierung der Krümmungen in einen horizontalen k-Bereich
                if min_k + i / anzahl_bereiche * wertebereich_k <= kruemmungen[j, 1] < min_k + (
                        i + 1) / anzahl_bereiche * wertebereich_k:
                    bereich = np.vstack((bereich, kruemmungen[j]))
            else:
                # Letzter Bereich einschließlich des Punkts mit maximaler Krümmung
                if min_k + i / anzahl_bereiche * wertebereich_k <= kruemmungen[j, 1] <= min_k + (
                        i + 1) / anzahl_bereiche * wertebereich_k:
                    bereich = np.vstack((bereich, kruemmungen[j]))
            j += 1
        j = 1
        while j < np.shape(bereich)[0]:
            # Ersten Punkt des Bereiches als ersten Punkt des Abschnittes wählen
            punktmenge = bereich[j - 1]
            # Abschnitt erweitern, solange der l-Abstand der Punkte < abstand_neues_be * median_l
            while j < np.shape(bereich)[0] and bereich[j][0] - bereich[j - 1,0] < abstand_neues_be* median_l:
                punktmenge = np.vstack((punktmenge, bereich[j]))
                j += 1
            # Falls entstandener Bereich größer als 8 Punkte ist: Beginn und Ende zum Array Abschnitte anfügen
            if np.shape(punktmenge)[0] > anzahl_pkte_parallelen:
                # Anfügen des Index in kruemmungen (nicht Wert der Länge)
                achsparallelen = np.vstack((achsparallelen, [int((np.where(kruemmungen[:, 0] == punktmenge[0][0])[0])),int((np.where(kruemmungen[:, 0] == punktmenge[-1][0])[0]))]))
            j += 1
        i += 1

    # Array sortieren und überlappende Intervalle zusammenfügen
    achsparallelen = achsparallelen[achsparallelen[:, 0].argsort()]
    startpunkte = achsparallelen[:, 0]
    endpunkte = np.maximum.accumulate(achsparallelen[:, 1])
    valid = np.zeros(len(achsparallelen) + 1, dtype=np.bool)
    valid[0] = True
    valid[-1] = True
    valid[1:-1] = startpunkte[1:] >= endpunkte[:-1]
    achsparallelen = np.vstack((startpunkte[:][valid[:-1]], endpunkte[:][valid[1:]])).T

    loeschen = np.empty([0,2])
    while True:
        if np.shape(loeschen)[0]>0:
            i = 0
            while i < np.shape(loeschen)[0]:
                index = np.where(achsparallelen[:,0] == loeschen[i, 0])[0]
                achsparallelen = np.delete(achsparallelen, index, 0)
                i += 1

        # Leere Matrix mit Geradenparametern
        c_achsparallelen = np.zeros(shape=(np.shape(achsparallelen)[0],6))
        i = 0
        while i < np.shape(c_achsparallelen)[0]:
            # Linke und rechte Grenze der benutzten Punkte gemäß Prozentsatz (aus den Indizes in achsparallelen)
            l = int(achsparallelen[i][0] + (achsparallelen[i][1] - achsparallelen[i][0] + 1) * (1 - prozentsatz_be)/2)
            r = int(achsparallelen[i][1] - (achsparallelen[i][1] - achsparallelen[i][0] + 1) * (1 - prozentsatz_be)/2)
            # Modellmatrix (Linearisierter Funktionaler Zusammenhang)
            X = np.concatenate((np.ones(shape=(r + 1 - l, 1)), nurlaengen[l:r + 1]), axis=1)
            # Beobachtungsvektor
            y = nurkruemmungen[l:r + 1]
            # Anzahl Messungen
            n = np.shape(y)[0]
            # Berechnung des Mittelwerts
            mittelwert = 1 / n * np.dot(np.ones(np.shape(np.transpose(y))), y)
            # Falls Mittelwert < kreis2gerade % des Medians der Krümmungen: Mittelwert = 0
            if abs(mittelwert) < kreis2gerade * median_k:
                mittelwert = 0
            c_achsparallelen[i][0] = mittelwert
            # Koeffizientenmatrix der Achsparallelen aktualisieren [Achsenabschnitt, Steigung=0, l_anfang, l_ende, Index_anfang, Index_ende]
            if achsparallelen[i, 0] == 0:
                c_achsparallelen[i, 2] = 0
            else:
                c_achsparallelen[i, 2] = nurlaengen[int(achsparallelen[i, 0])]
            c_achsparallelen[i, 3] = nurlaengen[int(achsparallelen[i, 1])]
            c_achsparallelen[i, 4] = int(achsparallelen[i, 0])
            c_achsparallelen[i, 5] = int(achsparallelen[i, 1])
            i += 1

        # Klothoidenmatrix erstellen
        klothoiden = np.zeros([np.shape(achsparallelen)[0] + 1, 4])
        # erste Zeile
        klothoiden[0, 1] = c_achsparallelen[0, 2]
        klothoiden[0, 2] = 0
        klothoiden[0, 3] = int(achsparallelen[0, 0])
        i = 1
        # Klothoidenarray füllen [laenge_anfang, laenge_nende, index_anfang, index_ende
        while i < np.shape(achsparallelen)[0]:
            klothoiden[i][0] = nurlaengen[int(achsparallelen[i - 1, 1])]
            klothoiden[i][1] = nurlaengen[int(achsparallelen[i, 0])]
            klothoiden[i][2] = int(achsparallelen[i - 1, 1])
            klothoiden[i][3] = int(achsparallelen[i, 0])
            i += 1
        # letzte Zeile (wie erste Zeile)
        klothoiden[-1][0] = nurlaengen[int(achsparallelen[-1, 1])]
        klothoiden[-1][1] = nurlaengen[-1]
        klothoiden[-1][2] = int(achsparallelen[-1, 1])
        klothoiden[-1][3] = np.where(nurlaengen == nurlaengen[-1])[0]

        # Klothoiden mit Länge null löschen
        i = 0
        while i < np.shape(klothoiden)[0]:
            if klothoiden[i][1] - klothoiden[i][0] == 0:
                klothoiden = np.delete(klothoiden, i, 0)
            i += 1

        # Koeffizienten der Klothoidengeraden (noch leer)
        c_klothoiden = np.empty([0, 2])
        # für jeden Klothoiden-Abschnitt
        i = 0
        while i < np.shape(klothoiden)[0]:
            # Grenzen aus Klothoiden-Array
            l = int(klothoiden[i, 2])
            r = int(klothoiden[i, 3])
            # prozentsatz_be nur anwenden, wenn genug übrigbleibt
            if (r + 1 - l) * (1 - prozentsatz_be) > 2:
                l = l + int((r + 1 - l ) * (1 - prozentsatz_be)/2)
                r = r - int((r + 1 - l ) * (1 - prozentsatz_be)/2)
            # Beobachtungsvektor
            y = nurkruemmungen[l:r + 1]
            # Funktionalmatrix zusammensetzen
            X = np.concatenate((np.ones(shape=(r+1-l,1)),nurlaengen[l:r+1]), axis=1)
            # Geschätzte Unbekannte
            cdach = ausgleichung(X, y)[0]
            # Neue Zeile an Koeffizientenmatrix der Klothoiden anhängen [k-achsenabschnitt, steigung]
            c_klothoiden = np.vstack((c_klothoiden, np.transpose(cdach)))
            i += 1
        # Arrays mergen: [k-achsenabschnitt, steigung, l_anfang, l_ende, index_anfang, index_ende]
        c_klothoiden = np.hstack((c_klothoiden, klothoiden))
        # Koeffizientenmatrizen der achsparallelen und klothoiden anhängen
        cEND = np.concatenate((c_klothoiden,c_achsparallelen), axis=0)
        # ... und nach l_anfang sortieren
        cEND = cEND[cEND[:, 2].argsort()]
        i = 0
        # l_anfang und l_ende durch Schnittpunkte aufeinanderfolgender BE ersetzen
        while i < np.shape(cEND)[0]-1:
            cEND[i, 3] = (cEND[i + 1, 0] - cEND[i, 0]) / (cEND[i, 1] - cEND[i + 1, 1])
            cEND[i + 1, 2] = cEND[i, 3]
            i += 1
        cEND[i, 3] = l_ges

        # Nulldurchgänge der Klothoiden
        i = 0
        while i < np.shape(cEND)[0]:
            # Wenn Klothoide
            if cEND[i, 1] != 0:
                # Wenn Nullstelle im Intervall (=Wendeklothoide)
                if cEND[i, 2] < -cEND[i, 0] / cEND[i, 1] < cEND[i, 3]:
                    # Grenzen übernehmen
                    l = int(cEND[i, 4])
                    r = int(cEND[i, 5])
                    # Falls mehr als zwei Punkte übrig bleiben, werden die Grenzen auf einen Bereich von prozent % der Punkte aktualisiert
                    #if (r + 1 - l) * (1 - prozentsatz_be) > 2:
                    #    l = l + int((r + 1 - l) * (1 - prozentsatz_be) / 2)
                    #    r = r - int((r + 1 - l) * (1 - prozentsatz_be) / 2)
                    # Aufteilen in positive und negative Punkte, zwei getrennte Geraden mit Schnittpunkt auf rechts-Achse
                    y_neg = np.empty([0, 1])
                    y_pos = y_neg
                    X_neg = np.empty([0, 2])
                    X_pos = X_neg
                    # Zähler startet bei linker Grenze
                    j = l
                    # Positive und negative Krümmungen trennen (X und y erstellen)
                    while j < r + 1:
                        if nurkruemmungen[j] < 0:
                            y_neg = np.vstack((y_neg, nurkruemmungen[j]))
                            X_neg = np.vstack((X_neg, np.hstack((1, nurlaengen[j]))))
                        else:
                            y_pos = np.vstack((y_pos, nurkruemmungen[j]))
                            X_pos = np.vstack((X_pos, np.hstack((1, nurlaengen[j]))))
                        j += 1
                    # Falls entweder bei negativer oder positiver Gerade nur ein Punkt zur Verfügung steht
                    if np.shape(y_neg)[0] == 0 or np.shape(y_pos)[0] == 0:
                        ls = - cEND[i, 0] / cEND[i, 1]
                        # Parameter zweimal speichern (für beide Geraden)
                        cEND = np.insert(cEND, i + 1, np.array((cEND[i, 0], cEND[i, 1], ls, cEND[i, 3], 0 , cEND[i,5])), 0)
                        cEND[i, 3] = ls
                        cEND[i, 5] = 0
                    # Sowohl für positive als auch für negative Gerade steht mehr als ein Punkt zur Verfügung
                    else:
                        # zukünftigen Schnittpunkt festlegen (Schnittpunkt Rechtsachse - Übergangspunkte +/-)
                        if X_neg[-1, 1] < X_pos[-1, 1]:
                            ls = (y_neg[-1]*X_pos[0,1]-y_pos[0]*X_neg[-1,1])/(y_neg[-1]-y_pos[0])
                        else:
                            ls = (y_neg[0]*X_pos[-1,1]-y_pos[-1]*X_neg[0,1])/(y_neg[0]-y_pos[-1])
                        # Neue X festlegen für Gerade mit (ls|0) als Zwangspunkt
                        X_neg = np.array_split(X_neg, [1], axis=1)[1] - ls
                        X_pos = np.array_split(X_pos, [1], axis=1)[1] - ls
                        # Parameter der neuen Geraden mit Schnittpunkt in (ls|0)
                        cdach_pos = np.array([-ausgleichung(X_pos, y_pos)[0][0] * ls, ausgleichung(X_pos, y_pos)[0][0]])
                        cdach_neg = np.array([-ausgleichung(X_neg, y_neg)[0][0] * ls, ausgleichung(X_neg, y_neg)[0][0]])
                        # Matrizen anpassen
                        if X_neg[-1]  < X_pos[-1] :
                            cdach_1 = cdach_neg
                            cdach_2 = cdach_pos
                        else:
                            cdach_1 = cdach_pos
                            cdach_2 = cdach_neg
                        # Einschneiden
                        if i == 0:
                            ls_vor = 0
                            ls_nach = (cEND[i + 1, 0]- cdach_2[0, 0])/(cdach_2[1, 0] - cEND[i+1, 1])
                            cEND[i + 1, 2] = ls_nach
                        elif i == np.shape(cEND)[0]-1:
                            ls_vor = (cEND[i - 1, 0]- cdach_1[0, 0])/(cdach_1[1, 0] - cEND[i-1, 1])
                            ls_nach = l_ges
                            cEND[i - 1, 3] = ls_vor
                        else:
                            ls_vor = (cEND[i - 1, 0]- cdach_1[0, 0])/(cdach_1[1, 0] - cEND[i-1, 1])
                            ls_nach = (cEND[i + 1, 0]- cdach_2[0, 0])/(cdach_2[1, 0] - cEND[i+1, 1])
                            cEND[i - 1, 3] = ls_vor
                            cEND[i + 1, 2] = ls_nach
                        # cEND-Matrix aktualisieren
                        cEND = np.insert(cEND, i + 1, np.array((cdach_2[0, 0], cdach_2[1, 0], ls, ls_nach, 0, cEND[i, 5])), 0)
                        cEND[i] = np.array([cdach_1[0, 0], cdach_1[1, 0], ls_vor, ls, cEND[i, 4], 0])
                i += 1
            i += 1

        i = 1
        loeschen = np.empty(shape=[0, 2])
        while i  < np.shape(cEND)[0] - 1:
            if cEND[i, 1] == 0 and cEND[i-1, 1] * cEND[i+1, 1] >= 0 and cEND[i, 3] - cEND[i, 2] < min_l_ei:
                loeschen = np.vstack((loeschen, np.array((cEND[i, 4], cEND[i, 5]))))
            i += 1
        if np.shape(loeschen)[0] == 0:
            break

    return cEND[:,:4]

def plotten(kruemmungen, cEND, anzahl_bereiche):
    """Plottet das Krümmungsbild mit lokalen Krümmungspunkten (blau), ausgleichenden Geraden (rot), Grenzen der Krümmungsbereiche (gelb)
       kruemmungen: Matrix mit Koordinaten (Länge, Krümmung),
       geradenarray: Matrix mit Geradenparametern"""

    # Alte Plots (falls vorhanden) schließen
    if plt.get_fignums():
        plt.close('all')

    # Vertikale Bereiche
    max_k = kruemmungen[:, 1].max()
    min_k = kruemmungen[:, 1].min()
    wertebereich_k = max_k - min_k
    i = 0
    while i <= anzahl_bereiche:
        plt.axhline(y=min_k + i * 1 / anzahl_bereiche * wertebereich_k, color='y')
        i += 1

    # Achsbeschriftungen und Achsen
    plt.xlabel("Länge [m]")
    plt.ylabel("Krümmung [1/km]")
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    # Rechts und Hoch gegeneinander abtragen
    x, y = zip(*kruemmungen)
    plt.scatter(x, y, marker='.')

    # Geraden eintragen
    i = 0
    while i < np.shape(cEND)[0]:
        x = np.linspace(cEND[i][2], cEND[i][3], 100)
        y = cEND[i][1] * x + cEND[i][0]
        plt.plot(x, y, 'r')
        i += 1
    # Fenster in die obere Linke Ecke setzen
    plt.get_current_fig_manager().window.setGeometry(50, 100, 600, 600)
    # Plot anzeigen
    plt.show()

class BogenElement:
    """Klasse, in deren Instanzen die Parameter der Bogenelemente gespeichert werden."""
    def __init__(self):
        self.laenge = 0.0
        self.ranf = 0.0
        self.rend = 0.0
        self.vz = 0.0
        self.art = 0.0
        self.A = 0.0
        self.xa = 0.0
        self.ya = 0.0
        self.riwia = 0.0
        self.xe = 0.0
        self.ye = 0.0
        self.riwie = 0.0
        self.stationa = "0+0.000"
        self.statione = "0+0.000"

    def gerade(self, xa, ya, riwia, laenge):
        """Berechnung von Zwischenpunkten auf Geraden"""
        riwie = riwia
        xe = xa + laenge * np.cos(riwia)
        ye = ya + laenge * np.sin(riwia)
        return riwie, xe, ye

    def kreis(self, xa, ya, riwia, laenge, radius, vz):
        """Berechnung von Zwischenpunkten auf Kreisbögen"""
        tau = laenge / radius
        Y = radius * np.sin(tau)
        X = radius - radius * np.cos(tau)
        sigma = 0
        if laenge > 0.0001:
            sigma = np.arctan2(X, Y)
        if sigma < 0:
            sigma = sigma + 2 * np.pi
        if vz == -1:
            Y = -Y
            sigma = 2 * np.pi - sigma
        s = np.sqrt(X ** 2 + Y ** 2)
        riwiS = riwia + sigma
        xe = xa + s * np.cos(riwiS)
        ye = ya + s * np.sin(riwiS)
        riwie = riwia + tau * vz
        return riwie, xe, ye

    def klotoide(self, laenge, A):
        """Berechnung von Zwischenpunkten auf Klothoiden mit Krümmung k = 0 im Anfangspunkt"""
        l = laenge / A
        y = l - l ** 5 / 40 + l ** 9 / 3456 - l ** 13 / 599040
        x = l ** 3 / 6 - l ** 7 / 336 + l ** 11 / 42240 - l ** 15 / 9676800
        X = A * x
        Y = A * y
        sigma = 0
        if laenge > 0.0001:
            sigma = np.arctan2(X, Y)
        if sigma < 0:
            sigma = sigma + 2 * np.pi
        s = np.sqrt(X ** 2 + Y ** 2)
        return sigma, s

    def klotoideA(self, xa, ya, riwia, laenge, A, vz):
        """Berechnung von Zwischenpunkten auf Klothoiden mit Krümmung k = 0 im Anfangspunkt"""
        tau = 0.5 * np.square(laenge) / np.square(A)
        sigma, s = self.klotoide(laenge, A)
        if vz == -1:
            sigma = 2 * np.pi - sigma
        riwiS = riwia + sigma
        xe = xa + s * np.cos(riwiS)
        ye = ya + s * np.sin(riwiS)
        riwie = riwia + tau * vz
        return riwie, xe, ye

    def klotoideE(self, xa, ya, riwia, b, A, vz, laenge):
        """Berechnung von Zwischenpunkten auf Klothoiden mit Krümmung k = 0 im Endpunkt"""
        tau_a = 0.5 * laenge ** 2 / A ** 2
        tau_e = 0.5 * (laenge - b) ** 2 / A ** 2
        sigma_a, s_a = self.klotoide(laenge, A)
        sigma_e, s_e = self.klotoide(laenge - b, A)
        riwie = riwia - vz * (tau_e - tau_a)
        xe = xa + s_a * np.cos(riwia + vz * (tau_a - sigma_a)) + s_e * np.cos(riwia - np.pi + vz * (tau_a - sigma_e))
        ye = ya + s_a * np.sin(riwia + vz * (tau_a - sigma_a)) + s_e * np.sin(riwia - np.pi + vz * (tau_a - sigma_e))
        return riwie, xe, ye

    def klotoideT(self, xa, ya, riwia, laenge, ranf, rend, vz, b):
        """Berechnung von Zwischenpunkten auf Eiklothoiden"""
        A = np.sqrt(ranf * rend * laenge / abs(ranf - rend))
        La = A ** 2 / ranf
        riwi_zurueck = riwia + np.pi  # entgegengesetzter Richtungswinke im Hauptpunkt
        vz_zurueck = -vz  # entgegengesetztes Vorzeichen im Hauptpunkt
        if ranf > rend:
            Le = La + b
            riwiaa, xaa, yaa = self.klotoideE(xa, ya, riwi_zurueck, La, A, vz_zurueck, La)
            riwiaa += np.pi
            riwie, xe, ye = self.klotoideA(xaa, yaa, riwiaa, Le, A, vz)
        else:
            Le = La - b
            riwiaa, xaa, yaa = self.klotoideE(xa, ya, riwia, La, A, vz, La)
            riwiaa = riwiaa + np.pi
            riwie, xe, ye = self.klotoideA(xaa, yaa, riwiaa, Le, A, vz_zurueck)
            riwie += np.pi
        return riwie, xe, ye

    def erstellen(self, c, punkte):
        """Array mit Instanzen vom Typ BogenElement aus der Matrix c, die die Geradenparameter aus dem Krümmungsbild enthält, erstellen.
            c = cEND (k-Achsenabschnitt, Steigung, xa, xe)
            punkte = Array mit digitalisierten Punkten"""
        nZeile = np.shape(c)[0]
        elemente = [BogenElement()] * nZeile
        for nr in range(nZeile):
            elemente[nr] = BogenElement()
            elemente[nr].laenge = c[nr, 3] - c[nr, 2] # Laenge Bogenstueck
            ka = (c[nr, 1] * c[nr, 2] + c[nr, 0])
            ke = (c[nr, 1] * c[nr, 3] + c[nr, 0])
            if ka == 0:
                elemente[nr].ranf = 0
            else:
                elemente[nr].ranf = abs(1000 / ka)  # Radius am Anfang
            if ke == 0:
                elemente[nr].rend = 0
            else:
                elemente[nr].rend = abs(1000 / ke)  # Radius am Ende
            if ka < 0 or ke < 0:
                elemente[nr].vz = 1
            elif ka > 0 or ke > 0:
                elemente[nr].vz = -1
            if ka == 0 and ka == ke: # 0=Gerade, 1=Kreis, 2=Anfangsklothoide,% 3=Endklothoide, 4=Teilklothoide
                elemente[nr].art = 0
            elif ka != 0 and ka == ke:
                elemente[nr].art = 1
            elif ka == 0 and ke != 0:
                elemente[nr].art = 2
            elif ka != 0 and ke == 0:
                elemente[nr].art = 3
            else:
                elemente[nr].art = 4

            if elemente[nr].art == 0 or elemente[nr].art == 1:
                elemente[nr].A = 0
            else:
                elemente[nr].A = np.sqrt(max(elemente[nr].ranf, elemente[nr].rend) * elemente[nr].laenge)  # Klotoidenparameter (Bei Kreis und Gerade=0)
        elemente[0].xa = punkte[0, 0]  # Koordinaten des Anfangspunkts
        elemente[0].ya = punkte[0, 1]
        elemente[0].riwia = riwi(punkte[0, 1], punkte[4, 1], punkte[0, 0], punkte[4, 0])  # Anfangsrichtung
        return elemente

def elemente_aktualisieren(elemente, stationierung):
    """Array mit BogenElementen wird aktualisiert (Durchlauf der Bogenfolge und Aktualisierung der Hauptpunkte)
        elemente: Array mit Instanzen vom Typ BogenElement (Bogenfolge)
        stationierung: 100 = Hektometrierung, 1000 = Kilometrierung"""

    if stationierung == 1000:
        formatPnr = "{0:06.2f}"
    else: # stationierung == 100
        formatPnr = "{0:05.2f}"

    geslaenge = 0
    nZeile = len(elemente)
    for nr in range(nZeile):
        elemente[nr].stationa = str(int(np.fix(geslaenge / stationierung))) + "+" + formatPnr.format(
            geslaenge - np.fix(geslaenge / stationierung) * stationierung)
        if elemente[nr].art == 0:
            elemente[nr].riwie, elemente[nr].xe, elemente[nr].ye = elemente[nr].gerade(
                elemente[nr].xa, elemente[nr].ya, elemente[nr].riwia, elemente[nr].laenge)
        elif elemente[nr].art == 1:
            elemente[nr].riwie, elemente[nr].xe, elemente[nr].ye = elemente[nr].kreis(
                elemente[nr].xa, elemente[nr].ya, elemente[nr].riwia, elemente[nr].laenge,
                elemente[nr].ranf, elemente[nr].vz)
        elif elemente[nr].art == 2:
            elemente[nr].riwie, elemente[nr].xe, elemente[nr].ye = elemente[nr].klotoideA(
                elemente[nr].xa, elemente[nr].ya, elemente[nr].riwia, elemente[nr].laenge,
                elemente[nr].A, elemente[nr].vz)
        elif elemente[nr].art == 3:
            elemente[nr].riwie, elemente[nr].xe, elemente[nr].ye = elemente[nr].klotoideE(
                elemente[nr].xa, elemente[nr].ya, elemente[nr].riwia, elemente[nr].laenge,
                elemente[nr].A, elemente[nr].vz, elemente[nr].laenge)
        elif elemente[nr].art == 4:
            elemente[nr].riwie, elemente[nr].xe, elemente[nr].ye = elemente[nr].klotoideT(
                elemente[nr].xa, elemente[nr].ya, elemente[nr].riwia, elemente[nr].laenge,
                elemente[nr].ranf, elemente[nr].rend, elemente[nr].vz, elemente[nr].laenge)
        if nr < nZeile - 1:
            elemente[nr + 1].xa = elemente[nr].xe
            elemente[nr + 1].ya = elemente[nr].ye
            elemente[nr + 1].riwia = elemente[nr].riwie
        geslaenge += elemente[nr].laenge
        elemente[nr].statione = str(int(np.fix(geslaenge / stationierung))) + "+" + formatPnr.format(
            geslaenge - np.fix(geslaenge / stationierung) * stationierung)
    return

def trassenpunkte_berechnen(elemente, zwlaenge, stationierung):
    """Berechnung der Bogenhaupt- und Bogenzwischenpunkte in Liste mit Instanzen der Klasse Trassenpunkt
        elemente: Liste mit Instanzen vom Typ BogenElement (Bogenfolge)
        zwlaenge: Abstand der Zwischenpunkte
        stationierung: 100 = Hektometrierung, 1000 = Kilometrierung"""

    if stationierung == 1000:
        formatPnr = "{0:06.2f}"
    else: # stationierung == 100
        formatPnr = "{0:05.2f}"

    nZeile = len(elemente)
    pktanzahl = 0
    geslaenge = 0
    for i in range(nZeile):
        geslaenge += elemente[i].laenge
        if geslaenge % zwlaenge != 0:
            pktanzahl += 1
    pktanzahl += int(geslaenge / zwlaenge + 1)
    trassenpunkte = [Trassenpunkt()] * pktanzahl
    pktnr = 0

    anfangsstueck = 0
    geslaenge = 0

    trassenpunkte[pktnr] = Trassenpunkt()
    trassenpunkte[pktnr].pnr = str(int(np.fix(geslaenge / stationierung))) + "+" + formatPnr.format(
        geslaenge - np.fix(geslaenge / stationierung) * stationierung)
    trassenpunkte[pktnr].x = elemente[0].xa
    trassenpunkte[pktnr].y = elemente[0].ya
    trassenpunkte[pktnr].schluessel = 1
    pktnr += 1

    for nr in range(nZeile):

        if anfangsstueck == 0.0:
            summe = zwlaenge
        else:
            summe = anfangsstueck

        while summe < elemente[nr].laenge:
            if elemente[nr].art == 0:
                riwi, xa, ya = elemente[nr].gerade(elemente[nr].xa, elemente[nr].ya,
                                                          elemente[nr].riwia, summe)
            elif elemente[nr].art == 1:
                riwi, xa, ya = elemente[nr].kreis(elemente[nr].xa, elemente[nr].ya,
                                                         elemente[nr].riwia, summe, elemente[nr].ranf,
                                                         elemente[nr].vz)
            elif elemente[nr].art == 2:
                riwi, xa, ya = elemente[nr].klotoideA(elemente[nr].xa, elemente[nr].ya,
                                                             elemente[nr].riwia, summe, elemente[nr].A,
                                                             elemente[nr].vz)
            elif elemente[nr].art == 3:
                riwi, xa, ya = elemente[nr].klotoideE(elemente[nr].xa, elemente[nr].ya,
                                                             elemente[nr].riwia, summe, elemente[nr].A,
                                                             elemente[nr].vz, elemente[nr].laenge)
            else: # art == 4
                riwi, xa, ya = elemente[nr].klotoideT(elemente[nr].xa, elemente[nr].ya,
                                                             elemente[nr].riwia, elemente[nr].laenge,
                                                             elemente[nr].ranf, elemente[nr].rend,
                                                             elemente[nr].vz, summe)
            geslaenge = geslaenge + summe
            trassenpunkte[pktnr] = Trassenpunkt()
            trassenpunkte[pktnr].pnr = str(int(np.fix(geslaenge / stationierung))) + "+" + formatPnr.format(
                geslaenge - np.fix(geslaenge / stationierung) * stationierung)
            trassenpunkte[pktnr].x = xa
            trassenpunkte[pktnr].y = ya
            trassenpunkte[pktnr].schluessel = 0
            geslaenge = geslaenge - summe
            summe = summe + zwlaenge
            pktnr += 1

        reststueck = elemente[nr].laenge - (summe - zwlaenge)
        anfangsstueck = zwlaenge - reststueck
        geslaenge = geslaenge + elemente[nr].laenge
        trassenpunkte[pktnr] = Trassenpunkt()
        trassenpunkte[pktnr].pnr = str(int(np.fix(geslaenge / stationierung))) + "+" + formatPnr.format(
            geslaenge - np.fix(geslaenge / stationierung) * stationierung)
        trassenpunkte[pktnr].x = elemente[nr].xe
        trassenpunkte[pktnr].y = elemente[nr].ye
        trassenpunkte[pktnr].schluessel = 1
        pktnr += 1
    return trassenpunkte

class Trassenpunkt:
    """Klasse, in deren Instanzen die Punkte der berechneten Trasse gespeichert werden."""
    def __init__(self):
        self.pnr = "0+000.00"
        self.x = 0.0
        self.y = 0.0
        self.schluessel = 0

def drehstreckung(layerpunkte, elemente, stationierung):
    """Drehstreckung der Bogenfolge auf den Endpunkt, alle Hauptpunkte und Richtungen werden aktualisiert.
        layerpunkte: Array mit digitalisierten Punkten
        elemente: iste mit Instanzen vom Typ BogenElement (Bogenfolge)
        stationierung: 100 = Hektometrierung, 1000 = Kilometrierung"""
    # Helmerttrafo (Drehstreckung) über Anfangs- und Endpunkt, Anbringen der Parameter an die Längen und Radien in c
    m = strecke(layerpunkte[0, 0], layerpunkte[-1, 0], layerpunkte[0, 1], layerpunkte[-1, 1]) / strecke(elemente[0].xa, elemente[-1].xe, elemente[0].ya, elemente[-1].ye)
    for element in elemente:
        element.ranf *= m
        element.rend *= m
        element.laenge *= m
        element.A *= m
    elemente_aktualisieren(elemente, stationierung)
    epsilon = riwi(layerpunkte[0, 0], layerpunkte[-1, 0], layerpunkte[0, 1], layerpunkte[-1, 1]) - riwi(
        elemente[0].xa, elemente[-1].xe, elemente[0].ya, elemente[-1].ye)
    elemente[0].riwia -= epsilon
    elemente_aktualisieren(elemente, stationierung)
    return

def achsliste(elemente):
    """Ausgabe der formatierten Achsliste auf der Python-Konsole von QGIS
        elemente: iste mit Instanzen vom Typ BogenElement (Bogenfolge)"""

    i = 0
    print("%9s %9s %10s %8s %8s %7s %14s %11s %12s %11s %12s" % (
    "Station A", "Station E", "Art", "Radius A", "Radius E", "A", "Krümmung", "East A", "North A", "East E", "North E"))
    while i < len(elemente):
        if elemente[i].art == 0:
            typ = "Gerade"
        elif elemente[i].art == 1:
            typ = "Kreisbogen"
        else:
            typ = "Klothoide"
        k_richtung = "---"
        if elemente[i].vz == 1:
            k_richtung = "linksgekrümmt"
        elif elemente[i].vz == -1:
            k_richtung = "rechtsgekrümmt"
        print("%9s %9s %10s %8.4f %8.4f %7.4f %14s %5.4f %5.4f %5.4f %5.4f" % (
        elemente[i].stationa, elemente[i].statione, typ, elemente[i].ranf, elemente[i].rend, elemente[i].A, k_richtung,
        elemente[i].xa, elemente[i].ya, elemente[i].xe, elemente[i].ye))
        i += 1
    return

# |||||||||||||||||||||||||||||||||||||||||||||||||| Plugin-Klasse |||||||||||||||||||||||||||||||||||||||||||||||||||||

class Freihandtrasse:
    """Plugin-Klasse"""

    # -------------------------------------------------------------------------------------------------
    # --------------------------------------- Konstruktor ---------------------------------------------
    # -------------------------------------------------------------------------------------------------

    def __init__(self, iface):
        """Konstruktor"""

        # Referenz zum Qgis-Interface
        self.iface = iface

    # -------------------------------------------------------------------------------------------------
    # -------------------------------- Plugin einladen und entladen -----------------------------------
    # -------------------------------------------------------------------------------------------------

    def initGui(self):
        """GUI intitialisieren (Menü-Einträge und Icons in der Toolbar erstellen)"""

        # Action, um Plugin-Konfiguration zu starten
        self.action = QAction(QIcon(":/plugins/Freihandtrasse/icon.png"), "Freihandtrasse",
                              self.iface.mainWindow())
        # Mit run-Methode verknüpfen
        self.action.triggered.connect(self.run)
        # Menü-Eintrag erstellen
        self.iface.addPluginToMenu("&Freihandtrasse", self.action)

    def unload(self):
        """Plugin-Menü und Icons entfernen"""
        self.iface.removePluginMenu("&Freihandtrasse", self.action)

    # -------------------------------------------------------------------------------------------------
    # ----------------------------------- Programmablauf über Gui -------------------------------------
    # -------------------------------------------------------------------------------------------------

    def programmGui(self):

        # ---------------------------------- Methoden -------------------------------------------------

        def digitalisierungstarten():
            # Geometrietyp und CRS setzen
            inputLayer = QgsVectorLayer('Point?crs=epsg:25832', 'Digitalisierung', 'memory')
            # neuen Layer zum Panel hinzufügen
            QgsProject.instance().addMapLayers([inputLayer])
            # Bearbeitungsmodus starten
            inputLayer.startEditing()
            # Feature-hinzufügen-Tool aktivieren
            self.iface.actionAddFeature().trigger()

        # -----------------------------------------------------------------------------------------

        def digitalisierungbeenden():
            # Diese Funktion wird aufgerufen, wenn der Button zum Beenden der Digitalisierung gedrückt wurde
            # inPutlayer ermitteln
            inputLayer = QgsProject.instance().mapLayersByName('Digitalisierung')[0]
            # Änderungen speichern und Bearbeitungsmodus beenden
            inputLayer.commitChanges()

        # -----------------------------------------------------------------------------------------

        def berechnen():

            # Layer auswählen, der zur Berechnung herangezogen wird
            if self.dlg.import_RB.isChecked():
                # Layer aus layerComboBox wählen
                import_layer_name = self.dlg.layer_CB.currentText()
                import_layer = QgsProject.instance().mapLayersByName(import_layer_name)[0]

                # Falls kein Punktlayer: Methode verlassen
                if not isinstance(import_layer, QgsVectorLayer) or import_layer.geometryType() != QgsWkbTypes.PointGeometry:
                    # Warnfahne erstellen
                    self.iface.messageBar().pushMessage("Fehler", "Bitte Layer mit Punktgeometrien wählen!",
                                                        level=Qgis.Critical, duration=3)
                    return

            # Selbst digitalisieren
            else:
                # Prüfen, ob Layer 'Digitalisierung' existiert
                if QgsProject.instance().mapLayersByName('Digitalisierung'):
                    # Layer 'Digitalisierung_temp' wählen
                    import_layer = QgsProject.instance().mapLayersByName('Digitalisierung')[0]
                    # Bearbeitungsmodus beenden
                    import_layer.commitChanges()
                else:
                    # Warnfahne erstellen
                    self.iface.messageBar().pushMessage("Fehler", "Bitte zuerst Punkte digitalisieren!",
                                                        level=Qgis.Critical, duration=3)
                    # Untermethode verlassen
                    return

            if QgsProject.instance().mapLayersByName('Trassenpunkte'):
                if import_layer.name() != 'Trassenpunkte':
                    # Layer löschen
                    zu_loeschen = QgsProject.instance().mapLayersByName('Trassenpunkte')
                    for layer in zu_loeschen:
                        QgsProject.instance().removeMapLayer(layer.id())

            # Layer in Array
            layerpunkte = layer2array(import_layer)

            # Krümmungen berechnen
            if self.dlg.npunkte_RB.isChecked():
                # glatt_k-ter Nachbarpunkt wird zur Berechnung der Kruemmung hinzugezogen
                glatt_k = self.dlg.npunkte_CB.currentIndex() + 1
                # Methode zum Berechnen der Krümmungen aufrufen
                try:
                    kruemmungen = kruemmungberechnen1(glatt_k, layerpunkte)
                except Exception:
                    self.iface.messageBar().pushMessage("Fehler", "Zu wenige Punkte für gewählte Glättung!",
                                                        level=Qgis.Critical, duration=3)
                    # Untermethode verlassen
                    return
            else:
                # Glättungsparameter abfragen (nut ungerade Zahlen zulassen)
                glatt_t = self.dlg.glatt_t_CB.currentIndex() * 2 + 1
                glatt_k = self.dlg.glatt_k_CB.currentIndex() * 2 + 1
                # Methode zum Berechnen der Krümmungen aufrufen
                try:
                    kruemmungen = kruemmungberechnen2(glatt_t, glatt_k, layerpunkte)
                except Exception:
                    self.iface.messageBar().pushMessage("Fehler", "Zu wenige Punkte für gewählte Glättung!",
                                                        level=Qgis.Critical, duration=3)
                    # Untermethode verlassen
                    return

            # Anzahl horizontaler k-Bereiche
            anzahl_bereiche = self.dlg.anzahl_bereiche_CB.currentIndex() + 1


            # min. Punktanzahl für Achsparallelen
            anzahl_pkte_parallelen = int(self.dlg.anzahl_pkte_parallelen_LE.text())
            # min. Länge eines Kreisbogens zwischen gleichsinnig gekrümmten Klothoiden
            min_l_ei = float(self.dlg.min_l_ei_LE.text())
            # Benutzte Punkte je Bogenelement
            prozentsatz_be = float(self.dlg.prozentsatz_be_LE.text()) / 100
            # kreis2gerade
            kreis2gerade = float(self.dlg.kreis2gerade_LE.text()) / 100
            # Neues BE im gleichen k-Bereich
            abstand_neues_be = float(self.dlg.abstand_neues_be_LE.text())
            # Abstand Bogenzwischenpunkte
            stationierung_zp = float(self.dlg.stationierung_zp_LE.text())
            # Format Stationierung
            if self.dlg.stationierung_CB.currentIndex() == 0:
                stationierung = 1000
            else: # self.dlg.stationierung_CB.currentIndex() == 1:
                stationierung = 100

            lges = 0
            i = 0
            while i + 1 < np.shape(layerpunkte)[0]:
                lges += strecke(layerpunkte[i][0], layerpunkte[i + 1][0], layerpunkte[i][1], layerpunkte[i + 1][1])
                i += 1


            if anzahl_pkte_parallelen < 3 or anzahl_pkte_parallelen > np.shape(kruemmungen)[0] or \
                    prozentsatz_be <= 0 or prozentsatz_be > 1 or \
                    min_l_ei < 0 or \
                    kreis2gerade <= 0 or kreis2gerade > 1 or \
                    abstand_neues_be < 1 or abstand_neues_be > np.shape(kruemmungen)[0] or \
                    stationierung_zp <= 0 or stationierung_zp > lges:

                self.iface.messageBar().pushMessage("Fehler", "Keine Berechnung möglich! Ungültige Angaben im Formular!",
                                                    level=Qgis.Critical, duration=3)
                return

            
            try:
                cEND = ausgleichende_geraden_2(kruemmungen, anzahl_pkte_parallelen, prozentsatz_be, anzahl_bereiche,
                                            kreis2gerade, abstand_neues_be, lges, min_l_ei)
            except Exception:
                self.iface.messageBar().pushMessage("Fehler", "Keine Berechnung möglich! Parameter für Geradenberechnung anpassen!",
                                                    level=Qgis.Critical, duration=3)
                return

            plotten(kruemmungen, cEND, anzahl_bereiche)

            # Wandeln der Geradenparameter in Elemente, Drehstreckung und Berechnung der Zwischenpunkte
            elemente = BogenElement().erstellen(cEND, layerpunkte)

            for element in elemente:
                if element.laenge < 0:
                    self.iface.messageBar().pushMessage("Fehler",
                                                        "Negative Elementlängen erzeugt!",
                                                        level=Qgis.Critical, duration=3)
                    return

            # Methoden hauptpunkte und drehstreckung ändern implizit auch Array Elemente!
            elemente_aktualisieren(elemente, stationierung)
            if self.dlg.drehstreckung_CB.isChecked():
                drehstreckung(layerpunkte, elemente, stationierung)
            zwischenpunkte = trassenpunkte_berechnen(elemente, stationierung_zp, stationierung)
            achsliste(elemente)
            output_layer = punkte2layer(zwischenpunkte)

            return

        # ------------------------------- GUI intitialisieren ------------------------------------------

        # Dialogfenster erstellen
        self.dlg = Freihandtrasse_Dialog()
        # Radiobuttons als Default
        self.dlg.import_RB.setChecked(True)
        self.dlg.npunkte_RB.setChecked(True)
        self.dlg.drehstreckung_CB.setChecked(True)
        # Aktuell geladene Layer
        layer = QgsProject.instance().layerTreeRoot().children()
        # Comboboxen leeren
        self.dlg.layer_CB.clear()
        self.dlg.npunkte_CB.clear()
        self.dlg.glatt_k_CB.clear()
        self.dlg.glatt_t_CB.clear()
        self.dlg.anzahl_bereiche_CB.clear()
        self.dlg.stationierung_CB.clear()
        # ComboBoxen füllen
        self.dlg.layer_CB.addItems([layer.name() for layer in layer])
        self.dlg.npunkte_CB.addItems([str(x + 1) for x in range(20)])
        self.dlg.glatt_k_CB.addItems([str(x + 1) for x in range(0, 20, 2)])
        self.dlg.glatt_t_CB.addItems([str(x + 1) for x in range(0, 20, 2)])
        self.dlg.anzahl_bereiche_CB.addItems([str(x + 1) for x in range(30)])
        self.dlg.stationierung_CB.addItems(["Kilometrierung", "Hektometrierung"])

        # Standard Input der LineEdit-Box zuweisen
        self.dlg.anzahl_pkte_parallelen_LE.setText("8")
        self.dlg.prozentsatz_be_LE.setText("80")
        self.dlg.kreis2gerade_LE.setText("10")
        self.dlg.abstand_neues_be_LE.setText("20")
        self.dlg.stationierung_zp_LE.setText("5")
        self.dlg.min_l_ei_LE.setText("10")
        self.dlg.anzahl_bereiche_CB.setCurrentIndex(14)
        self.dlg.npunkte_CB.setCurrentIndex(8)

        # Buttons mit Methoden verknüpfen
        self.dlg.berechnen_PB.clicked.connect(berechnen)
        self.dlg.digi_PB.clicked.connect(digitalisierungstarten)
        self.dlg.ende_digi_PB.clicked.connect(digitalisierungbeenden)

        # Dialogfenster anzeigen
        self.dlg.show()

        # Überprüfen, ob OK gedrückt wurde
        if self.dlg.exec_():
            # Überprüfen, ob der Layer 'Digitalisierung_temp' existiert
            if QgsProject.instance().mapLayersByName('Digitalisierung'):
                # Layer löschen
                to_be_deleted = QgsProject.instance().mapLayersByName('Digitalisierung')[0]
                QgsProject.instance().removeMapLayer(to_be_deleted.id())

    # -------------------------------------------------------------------------------------------------
    # ----------------------------------- Programmablauf über Steuerdatei -----------------------------
    # -------------------------------------------------------------------------------------------------

    def programmDatei(self):

        # steuerdatei.txt einlesen und in Variable steuerdatei schreiben
        try:
            file = open('steuerdatei.txt', 'r')
            steuerdatei = [file.read().splitlines()]
            file.close()
        except Exception:
            self.iface.messageBar().pushMessage("Fehler", "Steuerdatei defekt!",
                                                level=Qgis.Critical, duration=3)
            return
        i = 0

        while i < np.shape(steuerdatei[0])[0]:
            steuerdatei[0][i] = str(steuerdatei[0][i]).split(" ")[0]
            steuerdatei[0][i] = str(steuerdatei[0][i]).split("\t")[0]
            i += 1

        # Steuerparameter zuweisen
        import_layer = QgsProject.instance().mapLayersByName(steuerdatei[0][0])[0]
        artkruemmung = steuerdatei[0][1]
        glatt_t = int(steuerdatei[0][2])
        glatt_k = int(steuerdatei[0][3])
        anzahl_bereiche = int(steuerdatei[0][4])
        prozentsatz_be = float(steuerdatei[0][5]) / 100
        anzahl_pkte_parallelen = int(steuerdatei[0][6])
        kreis2gerade = float(steuerdatei[0][7]) / 100
        abstand_neues_be = int(steuerdatei[0][8])
        min_l_ei = float(steuerdatei[0][9])
        stationierung_zp = float(steuerdatei[0][10])
        stationierung = int(steuerdatei[0][11])

        # Layer in Array
        layerpunkte = layer2array(import_layer)

        lges = 0
        i = 0
        while i + 1 < np.shape(layerpunkte)[0]:
            lges += strecke(layerpunkte[i][0], layerpunkte[i + 1][0], layerpunkte[i][1], layerpunkte[i + 1][1])
            i += 1

        # Je nach Art der Berechnung Krümmungen berechnen
        # Exception-Handling nur bei Nutzer-Version, da Entwickler wissen sollten, welche Parameter unsinnig sind
        if artkruemmung == 'p':
            kruemmungen = kruemmungberechnen1(glatt_k, layerpunkte)
        else:
            kruemmungen = kruemmungberechnen2(glatt_t, glatt_k, layerpunkte)

        # Ausgleichende Geraden - Achsparallelen-Lösung
        try:
            cEND = ausgleichende_geraden_2(kruemmungen, anzahl_pkte_parallelen, prozentsatz_be, anzahl_bereiche, kreis2gerade, abstand_neues_be, lges, min_l_ei)
        except Exception:
            self.iface.messageBar().pushMessage("Fehler", "Keine Berechnung möglich! Bitte die Angaben im Formular anpassen!",
                                               level=Qgis.Critical, duration=3)
            return
        # Wandeln der Geradenparameter in Elemente, Drehstreckung und Berechnung der Zwischenpunkte
        elemente = BogenElement().erstellen(cEND, layerpunkte)
        for element in elemente:
            if element.laenge < 0:
                self.iface.messageBar().pushMessage("Fehler",
                                                    "Keine Berechnung möglich! Bitte höhere Glättung wählen!",
                                                    level=Qgis.Critical, duration=3)
                return
        # Methoden hauptpunkte und drehstreckung ändern implizit auch Array Elemente!
        elemente_aktualisieren(elemente, stationierung)
        drehstreckung(layerpunkte, elemente, stationierung)
        zwischenpunkte = trassenpunkte_berechnen(elemente, stationierung_zp, stationierung)
        output_layer = punkte2layer(zwischenpunkte)
        achsliste(elemente)

        # Plot mit Geraden
        plotten(kruemmungen, cEND, anzahl_bereiche)

    # ----------------------------------- Hauptprogramm-Methode ---------------------------------------

    def run(self):
        """Hauptmethode: Aufteilung in zwei Programmstränge"""
        if gui_erstellen:
            self.programmGui()
        else:
            self.programmDatei()
