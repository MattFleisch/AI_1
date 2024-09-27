import pandas as pd
import numpy as np

def get_data():

    fields = ["Accident_Severity", "Number_of_Vehicles","Number_of_Casualties", 
            "Day_of_Week", "Time",  "Road_Type", "Speed_limit", "Light_Conditions", 
            "Weather_Conditions", "Road_Surface_Conditions", "Urban_or_Rural_Area"]
    #time_cat = {"Morning": 0, "Afternoon": 0, "Evening": 0, "Night": 0}
    relationships = [#["Time", None],
                    ["Urban_or_Rural_Area", None],
                    ["Weather_Conditions", None],
                    ["Road_Type", "Urban_or_Rural_Area"],
                    ["Number_of_Vehicles", "Road_Type"],
                    ["Speed_limit", "Road_Type"],
                    ["Road_Surface_Conditions", "Weather_Conditions"],
                    ["Light_Conditions", "Weather_Conditions"], #"Time"],
                    ["Number_of_Casualties", "Number_of_Vehicles"],
                    ["Accident_Severity", "Speed_limit", "Road_Surface_Conditions", "Light_Conditions"]]
    
    prob_tables = {}    

    # Read in data
    df = pd.read_csv('accidents_2012_to_2014.csv', usecols=fields)

    for relation in relationships:

        if relation[1] != None:
            # Create a contingency table
            conditions = [df[relation[i]] for i in range(1, len(relation))]
            contingency_table = pd.crosstab(columns=df[relation[0]], index=conditions)

            # Calculate conditional probability
            row_totals = contingency_table.sum(axis=1)
            CPT = contingency_table.div(row_totals, axis=0)

            # Assign to dictionary
            cond_str = ", ".join(relation[1:])
            prob_tables[f"{relation[0]}|{cond_str}"] = CPT

        else:
            # Create a simple probability table (not conditional)
            value_counts = df[relation[0]].value_counts()
            
            # Calculate the probability table by dividing by the total count
            PT = value_counts / len(df)
            prob_tables[f"{relation[0]}|None"] = PT
            

    return prob_tables

get_data()