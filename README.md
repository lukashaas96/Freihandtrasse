# Freihandtrasse

- Abstract Summary

The topic of this bachelor’s thesis is the developing of a plugin for the free geographic information system QGIS used for the identification of elements of horizontal alignment of digitized freehand lines. In this context, a freehand line is to be understood as an apposition of digitized points of an already existing or intended route. Based on linear regression lines in an abraded curvature diagram, the route components are differentiated, the parameters are determined and issued. Concluding, the route can be recorded in a separate layer as main points of curve or intermedia points of curve in any random sta-tioning.

- Kurzzusammenfassung

Gegenstand der hier vorliegenden Bachelorarbeit ist die Entwicklung eines Plugins für das freie Geoinformationssystem QGIS zur Ermittlung von Trassierungselementen aus digitalisierten Freihandlinien. Unter einer Freihandlinie ist in diesem Zusammenhang eine Aneinanderreihung von digitalisierten Punkten einer bereits bestehenden oder geplanten Trasse zu verstehen. Auf der Grundlage von ausgleichenden Geraden im geglätteten Krümmungsbild werden die einzelnen Trassierungselemente abgegrenzt, deren Parameter ermittelt und diese ausgegeben. Abschließend kann die Trasse in Form von Bogenhaupt- und Bogenzwischenpunkten mit beliebig gewählter Stationierung in einem neuen Layer gespeichert werden. 

- Installation des Plugins

Um das Plugin als Nutzer verwenden zu können, sind einige Installationsschritte nötig.
Zuerst muss der gesamte Ordner "Freihandtrasse", der die in dieser Bachelorarbeit erstellten Dateien, die zur Verwendung des Plugins nötig sind, enthält, in das Verzeichnis der Plugins der QGIS-Installation kopiert werden. Das erstellte Plugin ist kompatibel mit allen zurzeit verfügbaren QGIS-Versionen ab dem Release 3.0. Seit diesem Release ist der Speicherort der Plugins unter folgendem Verzeichnis zu finden, in das der Ordner Freihandtrasse kopiert werden muss:

‘‘\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins‘

Der Ordner AppData befindet sich im Ordner des jeweiligen Benutzerprofils und ist meist ausgeblendet.
Nach einem Neustart von QGIS ist das Plugin im Kontextmenü [Erweiterungen] in der Auflistung aller Plugins unter dem Menüpunkt [Erweiterungen verwalten und installieren] gelistet. Hier existiert für jedes Plugin eine Informationsseite, die mit den Metadaten in der Datei metadata.txt gefüllt ist. Neben den Informationen zum Plugin und dessen Entwickler existiert hier die Möglichkeit, das Plugin zu installieren. Nach der Installation des Plugins erscheint im Kontextmenü [Erweiterungen] der Menüpunkt [Freihandtrasse]. Durch Auswahl des Menüpunktes wird das Plugin gestartet. 

Eine ausführliche Anleitung des Plugins, die auch auf die Parametrisierung eingeht, ist im Dokument "instructions" gegeben. 

- metadata

[general]
name=Freihandtrasse
email=lukas.haas@students.hs-mainz.de 
author=Lukas Haas
qgisMinimumVersion=3.0
description=Bestimmmung von Trassierungselementen aus in QGIS digitalisiertem grafischen Entwurf
about=Mithilfe dieses Plugins können Freihandlinien in Form von einzelnen Punkten eingelesen oder digitalisiert werden. Aus diesen Punkten errechnet das Plugin Trassierungselemente der Lage, um eine Achse an die digitalisierten Punkte annähern zu können. Nun können die Bogenhauptpunkte und Zwischenpunkte ausgegeben werden.
version=1.0

© 2020 Lukas Haas

Dieses Werk einschließlich seiner Teile ist urheberrechtlich geschützt. Jede Verwertung außerhalb der engen Grenzen des Urheberrechtgesetzes ist ohne Zustimmung des Autors unzulässig und strafbar. Das gilt insbesondere für Vervielfältigungen, Übersetzungen, Mikroverfilmungen sowie die Einspeicherung und Verarbeitung in elektronischen Systemen.
