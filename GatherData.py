import pandas as pd
import numpy as np

def get_data():

    fields = ["Accident_Severity", "Number_of_Vehicles","Number_of_Casualties", 
            "Day_of_Week", "Time",  "Road_Type", "Speed_limit", "Light_Conditions", 
            "Weather_Conditions", "Road_Surface_Conditions", "Urban_or_Rural_Area"]
    time_cat = {"Morning": 0, "Afternoon": 0, "Evening": 0, "Night": 0}
    results = {}

    # Read in data
    df = pd.read_csv('accidents_2012_to_2014.csv', usecols=fields)

    # Iterate over each field
    for field in fields:
        field_results = []
        #print("\n   Field: ", field)

        # Get probability list
        data = np.array(df[field].astype(str))
        unique, counts = np.unique(data, return_counts=True)
        probabilities = counts / counts.sum()

        # Print results
        if field != "Time":
            for value, probability in zip(unique, probabilities):
                if probability>=0.0001:
                    #print(f"Value: {value}, Probability: {probability:.4f}")
                    field_results.append({"Value:": value, "Probability:": probability})
        
        # Group time probabilities
        else:
            for value, probability in zip(unique, probabilities):
                if value[:2] != "na":
                    if 5 <= int(value[:2]) < 12:
                        category = "Morning"
                    elif 12 <= int(value[:2]) < 17:
                        category = "Afternoon"
                    elif 17 <= int(value[:2]) < 21:
                        category = "Evening"
                    else:
                        category = "Night"
                    time_cat[category] += probability
            for category, total_probability in time_cat.items():
                #print(f"Value: {category}, Probability: {total_probability:.4f}")
                field_results.append({"Value": category, "Probability": total_probability})
            
        results[field] = field_results
    
    return results