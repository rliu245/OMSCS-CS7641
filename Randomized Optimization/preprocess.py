# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:54:47 2019

@author: Ray
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import numpy as np

def PreProcessIncome(X, y):
    """
    Features:
        
    workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    fnlwgt: continuous.
    education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    education-num: continuous.
    marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    sex: Female, Male.
    capital-gain: continuous.
    capital-loss: continuous.
    hours-per-week: continuous.
    native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    """
    
    #workclass_onehotencoder = OneHotEncoder(sparse = False)
    #X['workclass'] = workclass_onehotencoder.fit_transform(np.array(X['workclass']).reshape(-1, 1))
    workclass_labelencoder = LabelEncoder()
    X['workclass'] = workclass_labelencoder.fit_transform(np.array(X['workclass']).reshape(-1, 1))
    
    fnlwgt_sc = StandardScaler()
    X['fnlwgt'] = fnlwgt_sc.fit_transform(np.array(X['fnlwgt']).reshape(-1, 1))
    
    #education_labelencoder = LabelEncoder()
    #X['education'] = education_labelencoder.fit_transform(np.array(X['education']).reshape(-1, 1))
    X = X.drop('education', axis =  1)
    
    #X.drop('education-num', axis = 1)
    
    #maritalStatus_onehotencoder = OneHotEncoder(sparse = False)
    #X['marital-status'] = maritalStatus_onehotencoder.fit_transform(np.array(X['marital-status']).reshape(-1, 1))
    maritalStatus_labelencoder = LabelEncoder()
    X['marital-status'] = maritalStatus_labelencoder.fit_transform(np.array(X['marital-status']).reshape(-1, 1))
    
    #occupation_onehotencoder = OneHotEncoder(sparse = False)
    #X['occupation'] = occupation_onehotencoder.fit_transform(np.array(X['occupation']).reshape(-1, 1))
    occupation_labelencoder = LabelEncoder()
    X['occupation'] = occupation_labelencoder.fit_transform(np.array(X['occupation']).reshape(-1, 1))
    
    #relationship_onehotencoder = OneHotEncoder(sparse = False)
    #X['relationship'] = relationship_onehotencoder.fit_transform(np.array(X['relationship']).reshape(-1, 1))
    relationship_labelencoder = LabelEncoder()
    X['relationship'] = relationship_labelencoder.fit_transform(np.array(X['relationship']).reshape(-1, 1))
    
    #race_onehotencoder = OneHotEncoder(sparse = False)
    #X['race'] = race_onehotencoder.fit_transform(np.array(X['race']).reshape(-1, 1))
    race_labelencoder = LabelEncoder()
    X['race'] = race_labelencoder.fit_transform(np.array(X['race']).reshape(-1, 1))
    
    #sex_onehotencoder = OneHotEncoder(sparse = False)
    #X['sex'] = sex_onehotencoder.fit_transform(np.array(X['sex']).reshape(-1, 1))
    sex_labelencoder = LabelEncoder()
    X['sex'] = sex_labelencoder.fit_transform(np.array(X['sex']).reshape(-1, 1))
    
    capitalGain_sc = StandardScaler()
    X['capital-gain'] = capitalGain_sc.fit_transform(np.array(X['capital-gain']).reshape(-1, 1))
    
    capitalLoss_sc = StandardScaler()
    X['capital-loss'] = capitalLoss_sc.fit_transform(np.array(X['capital-loss']).reshape(-1, 1))
    
    hoursPerWeek_sc = StandardScaler()
    X['hours-per-week'] = hoursPerWeek_sc.fit_transform(np.array(X['hours-per-week']).reshape(-1, 1))
    
    #nativeCountry_onehotencoder = OneHotEncoder(sparse = False)
    #X['native-country'] = nativeCountry_onehotencoder.fit_transform(np.array(X['native-country']).reshape(-1, 1))
    nativeCountry_labelencoder = LabelEncoder()
    X['native-country'] = nativeCountry_labelencoder.fit_transform(np.array(X['native-country']).reshape(-1, 1))
    
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(y)
    
    return X, y