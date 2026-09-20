import joblib
import pandas as pd
from fastapi import FastAPI 
from pydantic import BaseModel , Field
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware #this is both js,html,css form FastApi togethar
# Ex= backend run 5000port and frantend run 2200 port both port are connect by using CORS
            


model = joblib.load("Mental_Health_Model.pkl")
top_countries = ['Other','India','USA','Canada','Australia','UK' 'Germany', 'Mexico', 'Turkey', 'France']

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)



#A first Pydantic Model
class StudentData(BaseModel):

    Age : str = Field(... , gt=10 , le = 100)             
    Gender : Literal["Male" , "Female"]
    Country : str
    Academic_Level : Literal['Undergraduate', 'Graduate', 'High School']
    Most_Used_Platform : Literal['Facebook', 'LinkedIn', 'Instagram', 'Snapchat', 'Twitter','YouTube', 'TikTok', 'LINE', 'KakaoTalk', 'VKontakte', 'WhatsApp','WeChat']
    Purpose_Of_Use : Literal['Networking', 'Education', 'Entertainment', 'News']
    Avg_Daily_Usage_Hours : float = Field(..., ge=0 , le=24)
    Daily_Unlocks : int = Field(... , ge=0)
    Study_Hours : float = Field(..., ge = 0, le = 24)
    Physical_Activity_Hours : float = Field(... , ge = 0, le = 24)
    Sleep_Hours_Per_Night : float  = Field(... , ge = 0, le = 24 )
    Stress_Level : Literal['Medium', 'Low', 'Very High', 'High']
    
    
 
 # Describe whta we send back(respone body)
class PredictionResponse(BaseModel):
    predicted_mental_health_score:float  
    #6.777777 -> float
    

@app.get("/")
def greet():
    return {"Welcome to the side"}


@app.post("/predict" , response_model = PredictionResponse) #Show the output of the class 
def predict(data: StudentData):
    
    country_group = data.country if data.country in top_countries else "Other" 
    input_row = pd.DataFrame([{
        'Age':                          data.age,
        'Gender':                       data.gender,
        'Country':                      data.country,
        'Academic_Level':               data.academic_Level,
        'Most_Used_Platform':           data.most_used_platform,
        'Purpose_Of_Use':               data.purpose_of_use,
        'Avg_Daily_Usage_Hours':        data.avg_daily_usage_hours,
        'Daily_Unlocks':                data.daily_unlocks,
        'Study_Hours':                  data.study_hours,
        'Physical_Activity_Hours':      data.physical_activity_hours,
        'Sleep_Hours_Per_Night':        data.sleep_hours_per_night,
        'Stress_Level':                 data.stress_level,
        'Mental_Health_Score':          data.mental_health_score,
        'Grouped_country':              data.grouped_country
        
    }])
    
    prediction  = model.predict(input_row)[0]
    return PredictionResponse(predicted_mental_health_score=round(float(prediction),2))