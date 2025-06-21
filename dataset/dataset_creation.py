# ================================================
# Arquivo dataset-creation.py
#
# Descrição:
# - Este script realiza a criação dos Datasets BRFSS na versão mais atualizada (2023). 
# - Todo o código presente neste arquivo foi extraído e adaptado de Alex Teboul, portanto não é de autoria do grupo.
# - Utilizado apenas para gerar uma nova versão mais recente do Dataset para o projeto. 
# 
# Uso:
# Execute com Python 3.10 ou superior
#
# Observações:
# - Certifique-se de que o arquivo 'dataset_BRFSS2023.csv' esteja no mesmo diretório.
# - Os dados foram originalmente extraídos de um arquivo SAS (.xpt).
# ================================================

import os
import pandas as pd
import numpy as np
import random

#df = pd.read_sas('dataset.XPT', format='xport', encoding='utf-8')

#df.to_csv('dataset_BRFSS2023.csv', index=False)

random.seed(1)

brfss_2023_dataset = pd.read_csv('dataset_BRFSS2023.csv')

print("Forma do dataset completo (linhas, colunas):", brfss_2023_dataset.shape)

pd.set_option('display.max_columns', 500)

print("\nPrimeiras 5 linhas do dataset completo:")
print(brfss_2023_dataset.head())

colunas_selecionadas = [
    'DIABETE4', '_RFHYPE6', 'TOLDHI3', '_CHOLCH3', '_BMI5',
    'SMOKE100', 'CVDSTRK3', '_MICHD', '_TOTINDA', '_RFDRHV8',
    '_HLTHPL1', 'MEDCOST1', 'GENHLTH', 'MENTHLTH', 'PHYSHLTH',
    'DIFFWALK', '_SEX', '_AGEG5YR', 'EDUCA', 'INCOME3'
]

brfss_df_selected = brfss_2023_dataset[colunas_selecionadas]

print("\nForma do dataset filtrado (linhas, colunas):", brfss_df_selected.shape)

print("\nPrimeiras 5 linhas do dataset filtrado:")
print(brfss_df_selected.head())

#Drop Missing Values 
brfss_df_selected = brfss_df_selected.dropna()
brfss_df_selected.shape

# DIABETE4
# going to make this ordinal. 0 is for no diabetes or only during pregnancy, 1 is for pre-diabetes or borderline diabetes, 2 is for yes diabetes
# Remove all 7 (dont knows)
# Remove all 9 (refused)
brfss_df_selected['DIABETE4'] = brfss_df_selected['DIABETE4'].replace({2:0, 3:0, 1:2, 4:1})
brfss_df_selected = brfss_df_selected[brfss_df_selected.DIABETE4 != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.DIABETE4 != 9]
brfss_df_selected.DIABETE4.unique()

#1 _RFHYPE6
#Change 1 to 0 so it represetnts No high blood pressure and 2 to 1 so it represents high blood pressure
brfss_df_selected['_RFHYPE6'] = brfss_df_selected['_RFHYPE6'].replace({1:0, 2:1})
brfss_df_selected = brfss_df_selected[brfss_df_selected._RFHYPE6 != 9]
brfss_df_selected._RFHYPE6.unique()

#2 TOLDHI3
# Change 2 to 0 because it is No
# Remove all 7 (dont knows)
# Remove all 9 (refused)
brfss_df_selected['TOLDHI3'] = brfss_df_selected['TOLDHI3'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.TOLDHI3 != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.TOLDHI3 != 9]
brfss_df_selected.TOLDHI3.unique()

#3 _CHOLCH3
# Change 3 to 0 and 2 to 0 for Not checked cholesterol in past 5 years
# Remove 9
brfss_df_selected['_CHOLCH3'] = brfss_df_selected['_CHOLCH3'].replace({3:0,2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected._CHOLCH3 != 9]
brfss_df_selected._CHOLCH3.unique()

#4 _BMI5 (no changes, just note that these are BMI * 100. So for example a BMI of 4018 is really 40.18)
brfss_df_selected['_BMI5'] = brfss_df_selected['_BMI5'].div(100).round(0)
brfss_df_selected._BMI5.unique()

#5 SMOKE100
# Change 2 to 0 because it is No
# Remove all 7 (dont knows)
# Remove all 9 (refused)
brfss_df_selected['SMOKE100'] = brfss_df_selected['SMOKE100'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.SMOKE100 != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.SMOKE100 != 9]
brfss_df_selected.SMOKE100.unique()

#6 CVDSTRK3
# Change 2 to 0 because it is No
# Remove all 7 (dont knows)
# Remove all 9 (refused)
brfss_df_selected['CVDSTRK3'] = brfss_df_selected['CVDSTRK3'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.CVDSTRK3 != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.CVDSTRK3 != 9]
brfss_df_selected.CVDSTRK3.unique()

#7 _MICHD
#Change 2 to 0 because this means did not have MI or CHD
brfss_df_selected['_MICHD'] = brfss_df_selected['_MICHD'].replace({2: 0})
brfss_df_selected._MICHD.unique()

#8 _TOTINDA
# 1 for physical activity
# change 2 to 0 for no physical activity
# Remove all 9 (don't know/refused)
brfss_df_selected['_TOTINDA'] = brfss_df_selected['_TOTINDA'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected._TOTINDA != 9]
brfss_df_selected._TOTINDA.unique()

#9 _RFDRHV8
# Change 1 to 0 (1 was no for heavy drinking). change all 2 to 1 (2 was yes for heavy drinking)
# remove all dont knows and missing 9
brfss_df_selected['_RFDRHV8'] = brfss_df_selected['_RFDRHV8'].replace({1:0, 2:1})
brfss_df_selected = brfss_df_selected[brfss_df_selected._RFDRHV8 != 9]
brfss_df_selected._RFDRHV8.unique()

#10 _HLTHPL1
# 1 is yes, change 2 to 0 because it is No health care access
# remove 9 for don't know or refused
brfss_df_selected['_HLTHPL1'] = brfss_df_selected['_HLTHPL1'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected._HLTHPL1 != 9]
brfss_df_selected._HLTHPL1.unique()

#11 MEDCOST1
# Change 2 to 0 for no, 1 is already yes
# remove 7 for don/t know and 9 for refused
brfss_df_selected['MEDCOST1'] = brfss_df_selected['MEDCOST1'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.MEDCOST1 != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.MEDCOST1 != 9]
brfss_df_selected.MEDCOST1.unique()

#12 GENHLTH
# This is an ordinal variable that I want to keep (1 is Excellent -> 5 is Poor)
# Remove 7 and 9 for don't know and refused
brfss_df_selected = brfss_df_selected[brfss_df_selected.GENHLTH != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.GENHLTH != 9]
brfss_df_selected.GENHLTH.unique()

#13 MENTHLTH
# already in days so keep that, scale will be 0-30
# change 88 to 0 because it means none (no bad mental health days)
# remove 77 and 99 for don't know not sure and refused
brfss_df_selected['MENTHLTH'] = brfss_df_selected['MENTHLTH'].replace({88:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.MENTHLTH != 77]
brfss_df_selected = brfss_df_selected[brfss_df_selected.MENTHLTH != 99]
brfss_df_selected.MENTHLTH.unique()

#14 PHYSHLTH
# already in days so keep that, scale will be 0-30
# change 88 to 0 because it means none (no bad mental health days)
# remove 77 and 99 for don't know not sure and refused
brfss_df_selected['PHYSHLTH'] = brfss_df_selected['PHYSHLTH'].replace({88:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.PHYSHLTH != 77]
brfss_df_selected = brfss_df_selected[brfss_df_selected.PHYSHLTH != 99]
brfss_df_selected.PHYSHLTH.unique()

#15 DIFFWALK
# change 2 to 0 for no. 1 is already yes
# remove 7 and 9 for don't know not sure and refused
brfss_df_selected['DIFFWALK'] = brfss_df_selected['DIFFWALK'].replace({2:0})
brfss_df_selected = brfss_df_selected[brfss_df_selected.DIFFWALK != 7]
brfss_df_selected = brfss_df_selected[brfss_df_selected.DIFFWALK != 9]
brfss_df_selected.DIFFWALK.unique()

#16 _SEX
# in other words - is respondent male (somewhat arbitrarily chose this change because men are at higher risk for heart disease)
# change 2 to 0 (female as 0). Male is 1
brfss_df_selected['_SEX'] = brfss_df_selected['_SEX'].replace({2:0})
brfss_df_selected._SEX.unique()

#17 _AGEG5YR
# already ordinal. 1 is 18-24 all the way up to 13 wis 80 and older. 5 year increments.
# remove 14 because it is don't know or missing
brfss_df_selected = brfss_df_selected[brfss_df_selected._AGEG5YR != 14]
brfss_df_selected._AGEG5YR.unique()

#18 EDUCA
# This is already an ordinal variable with 1 being never attended school or kindergarten only up to 6 being college 4 years or more
# Scale here is 1-4
# Remove 9 for refused:
brfss_df_selected = brfss_df_selected[brfss_df_selected.EDUCA != 9]
brfss_df_selected.EDUCA.unique()

#19 INCOME3
# Variable is already ordinal with 1 being less than $10,000 all the way up to 8 being $75,000 or more
# Remove 77 and 99 for don't know and refused
brfss_df_selected = brfss_df_selected[brfss_df_selected.INCOME3 != 77]
brfss_df_selected = brfss_df_selected[brfss_df_selected.INCOME3 != 99]
brfss_df_selected.INCOME3.unique()

#Check the shape of the dataset 
print(brfss_df_selected.shape)

#Let's see what the data looks like after Modifying Values
print(brfss_df_selected.head())

#Check Class Sizes of the heart disease column
print(brfss_df_selected.groupby(['DIABETE4']).size())

#Rename the columns to make them more readable
brfss = brfss_df_selected.rename(columns = {'DIABETE4':'Diabetes', 
                                         '_RFHYPE6':'Pressão_Alta',  
                                         'TOLDHI3':'Colesterol_Alto', '_CHOLCH3':'Avaliou_Colesterol', 
                                         '_BMI5':'IMC', 
                                         'SMOKE100':'Fumante', 
                                         'CVDSTRK3':'Ataque_Cardíaco', '_MICHD':'Doença_Coronário_ouInfarto', 
                                         '_TOTINDA':'Atividade_Física', 
                                         '_RFDRHV8':'Consumo_Álcool', 
                                         '_HLTHPL1':'Seguro_Saúde', 'MEDCOST1':'Acesso_Saúde', 
                                         'GENHLTH':'Saúde_Geral', 'MENTHLTH':'Saúde_Mental', 'PHYSHLTH':'Saúde_Física', 'DIFFWALK':'Dificuldade_Andar', 
                                         '_SEX':'Gênero', '_AGEG5YR':'Idade', 'EDUCA':'Nível_Educação', 'INCOME3':'Renda' })

print(brfss.head())

print(brfss.shape)

#Check how many respondents have no diabetes, prediabetes or diabetes. Note the class imbalance!
print(brfss.groupby(['Diabetes']).size())

#************************************************************************************************
brfss.to_csv('diabetes_012_health_indicators_BRFSS2023.csv', sep=",", index=False)
#************************************************************************************************

#Copy old table to new one.
brfss_binary = brfss

#Change the diabetics 2 to a 1 and pre-diabetics 1 to a 0, so that we have 0 meaning non-diabetic and pre-diabetic and 1 meaning diabetic.
brfss_binary['Diabetes'] = brfss_binary['Diabetes'].replace({1:0})
brfss_binary['Diabetes'] = brfss_binary['Diabetes'].replace({2:1})

#Change the column name to Diabetes_binary
brfss_binary = brfss_binary.rename(columns = {'Diabetes': 'Diabetes_binário'})
brfss_binary.Diabetes_binary.unique()

#Show the change
print(brfss_binary.head())

#show class sizes
print(brfss_binary.groupby(['Diabetes_binário']).size())

#Separate the 0(No Diabetes) and 1&2(Pre-diabetes and Diabetes)
#Get the 1s
is1 = brfss_binary['Diabetes_binário'] == 1
brfss_5050_1 = brfss_binary[is1]

#Get the 0s
is0 = brfss_binary['Diabetes_binário'] == 0
brfss_5050_0 = brfss_binary[is0] 

#Select the 39977 random cases from the 0 (non-diabetes group). we already have 35346 cases from the diabetes risk group
brfss_5050_0_rand1 = brfss_5050_0.take(np.random.permutation(len(brfss_5050_0))[:35346])

#Append the 39977 1s to the 39977 randomly selected 0s
brfss_5050 = pd.concat([brfss_5050_0_rand1, brfss_5050_1], ignore_index=True)

#Check that it worked. Now we have a dataset of 79,954 rows that is equally balanced with 50% 1 and 50% 0 for the target variable Diabetes_binary
print(brfss_5050.head())

print(brfss_5050.tail())

#See the classes are perfectly balanced now
print(brfss_5050.groupby(['Diabetes_binário']).size())

print(f'brfss_5050={brfss_5050.shape}',f'brfss_binary={brfss_binary.shape}')

#Save the 50-50 balanced dataset to csv
#************************************************************************************************
brfss_5050.to_csv('diabetes_binary_5050split_health_indicators_BRFSS2023.csv', sep=",", index=False)
#************************************************************************************************

#Also save the original binary dataset to csv
#************************************************************************************************
brfss_binary.to_csv('diabetes_binary_health_indicators_BRFSS2023.csv', sep=",", index=False)
#************************************************************************************************

