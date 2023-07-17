import numpy as np

def evaluate_model(y_true, y_predicted):
    """
    Evaluiert die Performance des Modells auf den Trainings- und Validierungsdaten
    :param y_true, y_predicted: Tatsächliche Kurse und vorhergesagte Kurse
    :return total_score (MSE), total_mean_price (Durschnittskurs)
    """
    
    total_score = 0
    mean_price = 0
    abs_dif = 0
    #Berechne Kennzahlen:
    for row in range(y_true.shape[0]):
        for col in range(y_predicted.shape[1]): #Gehe jedes Element durch und berechne dafür die Kennzahl
            mean_price = mean_price + y_true[row, col]
            total_score = total_score + (y_true[row, col] - y_predicted[row, col])**2
            abs_dif = abs_dif + (abs(y_true[row, col] - y_predicted[row, col]))
    total_score = np.sqrt(total_score/(y_true.shape[0] * y_predicted.shape[1])) #Berechne Durchschnitt
    total_mean_price = mean_price/(y_true.shape[0] * y_predicted.shape[1])
    return total_score, total_mean_price