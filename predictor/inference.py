import lightgbm as lgb
import pandas as pd
import numpy as np

data_file_csv = "indian_crop_data_realistic_v2.csv"
data = pd.read_csv(data_file_csv)
df = pd.DataFrame(data)
data = df.iloc[[0]]
depth_model_path = "planting_depth_model_final.txt"
yield_predictor_path = "yield_predictor_final_v2.txt"

model_depth = lgb.Booster(model_file=depth_model_path)
model_yield = lgb.Booster(model_file=yield_predictor_path)

class prediction :
    def __init__(self):
        self.model_depth = model_depth
        self.model_yield = model_yield
        self.depth_target = ["Average_Planting_Depth_in_cm"]
        self.yield_target = ["Yield_kg_per_hectare"]
        self.cat = ["Soil_Type" , "Crop_Type"]
    def predict_depth(self , data):
        df = pd.DataFrame(data)
        X1 = df.drop(self.depth_target , axis=1)
        X1['Crop_Type'] = X1['Crop_Type'].astype('category')
        X1['Soil_Type'] = X1['Soil_Type'].astype('category')
        Y1 = df.loc[: , self.depth_target]
        
        prediction = self.model_depth.predict(X1)
        return prediction[0]
    
    def predict_yield(self , data):
        df = pd.DataFrame(data)
        X2 = df.drop(self.yield_target , axis=1)
        X2['Crop_Type'] = X2['Crop_Type'].astype('category')
        X2['Soil_Type'] = X2['Soil_Type'].astype('category')
        Y2 = df.loc[: , self.yield_target]
        
        prediction = self.model_yield.predict(X2)
        return prediction[0]
    
    def predict(self , data):
        depth = self.predict_depth(data)
        yields =  self.predict_yield(data)
        
        return {'depth':depth , 'yield':yields}
    
p = prediction()
result = p.predict(data)
print(f"The depth is : {result['depth']:.2f} cm\nThe Yield is : {result['yield']:.3f} kg/hectare")

#The prediction object can be imported and used anywhere , this is the complete module needed .