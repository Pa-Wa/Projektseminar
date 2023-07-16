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

Information zum Kommentieren: Beim Dashboard wurden sich wiederholende Elemente nicht erneut kommentiert. Demnach bitte von "vorne" nach "hinten" lesen, bzw. von app.py nach pg6.
