import numpy as np

def data_to_windowed_data(df, windows):
    """
    Wendet die Windowing-Technik an
    :param df: Dataframe der historischen Daten
    :param windows: Windowgröße
    :return: df (DF nach Anwendung der Windowing-Technik
    """
    
    close_prices = df["Close"]
    for i in range(windows - 1, -1, -1): #Geht alle Elemente des DF durch und erzeugt Windows
        column_name = f"Target_{i+1}"
        df.loc[:, column_name] = close_prices.shift(i+1) #Erzeugt Windows und speichert sie ab
    df.reset_index(inplace = True) #Index zu Spalte
    df.rename(columns = {"Date": "Target Date"}, inplace = True) #Umbenennung
    col = df.pop("Close") #Entferne Close-Spalte und füge sie mit anderem Namen an anderer Stelle ein
    df.insert(loc = len(df.columns), column = "Target", value = col) 
    df.dropna(inplace = True) #Entferne Nan's (erste Windowgröße-Elemente)
    return df

def windowed_df_to_d_x_y(wdf):
    """
    Umformatierung des Window-DF in bestimmtes Format für Tensorflow
    :param wdf: Windowed-DF
    :return: dates, X, Y (Daten, Input-Daten, Target-Daten)
    """
    
    df_as_np = wdf.to_numpy() #Zu numpy Format
    dates = df_as_np[:, 0] #Dates = 1.Spalte des DF
    pre_matrix = df_as_np[:, 1: -1] #Input-Daten (Window, ohne Target)
    X = pre_matrix.reshape((len(dates), pre_matrix.shape[1], 1)) #Umformatierung
    Y = df_as_np[:, -1] #Target-Daten
    return dates, X.astype(np.float32), Y.astype(np.float32)