# Finanz-Dashboard und Verfahren zur Kursprognose von Aktien/Indizes

Autoren: Yann Chevelaz und Pascal Wagener

Business Analytics Projektseminar Sommersemester 2023

Universität Siegen

---

#### Doku:
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
  * saved_model: Vortrainierte Modelle für die Methoden LSTM und LSTM-OS für die Aktien Tesla und Nvidia (Achtung: Training nicht UpToDate & Dash-Ordner als Pfad in Kommandozeile, damit Trainingsdaten gefunden werden!)

* Forecast:
  * evaluation_arch_param_LSTM_methods: Evaluation der besten Architektur/Parametereinstellungen der LSTM-Modelle
    * eval_LSTM
    * eval_LSTM_OS
    * other_func (für LSTM und LSTM_OS)
  * evaluation_forecast: Evaluierung der Performance der einzelnen Prognosemethoden anhand mehrerer Testsets
    * eval_forecast: Main-Datei (Anpassbar für jede Methode und verschiedene Analysen)
    * forecast_methods: Funktionen der einzelnen Prognosen (HW, ARIMA, LSTM, LSTM-OS, Naive)

* Unused (ohne Relevanz): Verworfene, bzw. in andere Dateien implementierte Elemente und Dateien zur Erstellung von Plots (nicht kommentiert) -> benötigt keiner Betrachtung

***Information zur Code-Kommentierung:*** Beim Dashboard wurden sich wiederholende Elemente nicht erneut kommentiert. Demnach bitte von "vorne" nach "hinten" lesen, bzw. von app.py nach pg6.
