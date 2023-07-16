# BuA Projektseminar SS23 (Chevalaz/Wagener)
Finanz-Dashboard und Verfahren zur Kursprognose von Aktien/Indizes

#### Dokumentenübersicht:
* Dash:
  * app.py: Main-Datei (auszuführende Datei) der Dash-App; Layout Definition der einzelnen Pages
  * pages:
    * pg1: Übersichts-Seite
    * pg2: Analyse-Seite
    * pg3: Holt-Winters Prognose
    * pg4: ARIMA Prognose
    * pg5: LSTM Prognose
    * pg6: LSTM One-Shot Prognose
  * methods: Funktionen der einzelnen Prognosen (HW, ARIMA, LSTM, LSTM-OS)
  * saved_models: Vortrainierte Modelle für die Methoden LSTM und LSTM-OS für die Aktien Tesla und Nvidia (Achtung: Training nicht UpToDate)

* Unused: 

Information zum Kommentieren: Darauf ausgelegt, dass Coding des Dashboards von Seite zu Seite gelesen wird. Sich wiederholende Elemente wurden kein 2. mal kommentiert außer sie weisen eine große relevantz auf
