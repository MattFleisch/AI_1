from pylab import *
import matplotlib.pyplot as plt
import os
import pyAgrum as gum
from TableGather import get_data
import pandas as pd

def string_int_lst(string_lst):

    # Check for unique entries
    unique_lst = []
    for string in string_lst:
        if string not in unique_lst:
            unique_lst.append(string)
        
    # Create a dictionary mapping each string to an integer starting from 0
    return {string: index for index, string in enumerate(unique_lst)}

def make_bn():

    relationships = [#["Time", None],
                    ["Urban_or_Rural_Area", "None"],
                    ["Weather_Conditions", "None"],
                    ["Road_Type", "Urban_or_Rural_Area"],
                    ["Number_of_Vehicles", "Road_Type"],
                    ["Speed_limit", "Road_Type"],
                    ["Road_Surface_Conditions", "Weather_Conditions"],
                    ["Light_Conditions", "Weather_Conditions"], #"Time"],
                    ["Number_of_Casualties", "Number_of_Vehicles"],
                    ["Accident_Severity", "Speed_limit", "Road_Surface_Conditions", "Light_Conditions"]]

    all_tables = get_data()
    bn=gum.BayesNet('CarAccidents')

    for relation in relationships:
        condition_str = ", ".join(relation[1:])
        CPT = all_tables[f"{relation[0]}|{condition_str}"]

        # Add nodes
        if relation[1] == "None":
            bn.add(gum.LabelizedVariable(relation[0], "", CPT.shape[0]))
        else:
            bn.add(gum.LabelizedVariable(relation[0], "", len(CPT.columns)))

        # Add arks    
        if relation[1] != "None":
            for i in range(1,len(relation)):
                bn.addArc(relation[i], relation[0])

        # Add PTs (Non-conditional probabilities)
        if relation[1] == "None":
            bn.cpt(relation[0]).fillWith(CPT.values)        

        # Add CPTs 
        else:
            # One condition
            if len(relation) == 2:
                for k in range(len(CPT.index)):
                    bn.cpt(relation[0])[{relation[1] : k}] = CPT.values[k].tolist()
            # Three conditions
            elif len(relation) == 4:
                dicts = [string_int_lst([index[i] for index in CPT.index]) for i in range(3)] #string to integer associations
                for l, index in enumerate(CPT.index):
                    bn.cpt(relation[0])[{relation[i + 1]: dicts[i].get(index[i]) for i in range(3)}] = CPT.values[l].tolist() #assign probability to each condition perputation
            
    return bn

def main():
    ie=gum.LazyPropagation(make_bn())

    # An Example
    ie.setEvidence({"Speed_limit":4})
    ie.makeInference()
    print(ie.posterior("Accident_Severity"))

if __name__ == "__main__":
    main()