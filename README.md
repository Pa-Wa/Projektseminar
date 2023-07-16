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
  * saved_model: Vortrainierte Modelle für die Methoden LSTM und LSTM-OS für die Aktien Tesla und Nvidia (Achtung: Training nicht UpToDate)

* Forecast:
  * Evaluation_Arch_Param_LSTM_Methods: Evaluation der besten Architektur/Parametereinstellungen der LSTM-Modelle
    * Eval_LSTM
    * Eval_LSTM_OS  
  * Evaluation_Prediction: Evaluierung der Performance der einzelnen Prognosemethoden anhand mehrerer Testsets
    * Eval_Prediction: Main-Datei (Anpassbar für jede Methode und verschiedene Analysen)
    * Naive: Naive Prognose
    * Holt-Winters: Holt-Winters Prognose
    * Arima: ARIMA Prognose
    * LSTM: LSTM Prognose
    * LSTM_OS: LSTM One-Shot Prognose

* Unused (ohne Relevanz): Verworfene, bzw. in andere Dateien implementierte Elemente und Dateien zur Erstellung von Plots (nicht kommentiert) -> benötigt keiner Betrachtung

***Information zum Kommentieren:*** Beim Dashboard wurden sich wiederholende Elemente nicht erneut kommentiert. Demnach bitte von "vorne" nach "hinten" lesen, bzw. von app.py nach pg6.
