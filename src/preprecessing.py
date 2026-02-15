import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_input(df: pd.DataFrame):
    df.drop(columns=['Name', 'PassengerId'], inplace=True)
   # Split Cabin into Deck, Number, Side
    df[['Deck','CabinNum','Side']] = df['Cabin'].str.split('/', expand=True)
    # Convert CabinNum to numeric
    df['CabinNum'] = pd.to_numeric(df['CabinNum'], errors='coerce')
    df.drop(columns=['Cabin'], inplace=True)
    df.drop(columns=['Deck'], inplace=True) 
    
    df['IsKid'] = df['Age'] < 18
    
    expense_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df["TotalExpenses"] = df[expense_cols].sum(axis=1)
    df["TotalExpenses"] = np.where(df["VIP"], df["TotalExpenses"] ** 2, df["TotalExpenses"])
    
    # Load saved objects
    ohe = joblib.load("Models/ohe.pkl")
    scaler = joblib.load("Models/scaler.pkl")
    train_columns = joblib.load("Models/columns.pkl")

    cat_cols = ['HomePlanet','Destination','Side']
    num_cols = ['Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck',
                'CabinNum','TotalExpenses','CryoSleep','VIP','IsKid']

    # Transform categorical
    encoded_cat = ohe.transform(df[cat_cols])
    encoded_cat_df = pd.DataFrame(
        encoded_cat,
        columns=ohe.get_feature_names_out(cat_cols)
    )

    # Scale numerical
    scaled_num = scaler.transform(df[num_cols])
    scaled_num_df = pd.DataFrame(scaled_num, columns=num_cols)

    # Combine
    df_processed = pd.concat([scaled_num_df, encoded_cat_df], axis=1)

    # Ensure exact same column order
    df_processed = df_processed.reindex(columns=train_columns, fill_value=0)

    return df_processed.to_numpy(dtype=np.float64)
    