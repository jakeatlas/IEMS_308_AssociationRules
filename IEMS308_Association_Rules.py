#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:21:10 2018

@author: jakeatlas
"""

#%% OBTAIN DATA FROM TRANSACTION TABLE & WRITE NEW TABLE IN jaa977_schema
# ******* Note: This section of the code may take over an hour to run and requires Northwestern VPN connection ********

import psycopg2
import pandas as pd
from sqlalchemy import create_engine 

#Connect to database
connection = psycopg2.connect(host = "gallery.iems.northwestern.edu", user = 'jaa977', password = 'jaa977_pw', dbname = 'iems308')

#Establish cursor
cur = connection.cursor()

#Obtain dataframe of Biloxi, MS transactions
cur.execute("SELECT * FROM pos.trnsact WHERE c2 = '4902'")
df_biloxi_transact = pd.DataFrame(cur.fetchall())
df_biloxi_transact.columns = ['sku','store','register','trannum','seq','saledate','stype','quantity','amt','orgprice','ignore_orgprice','interid','mic','ignore']

#Remove all columns except sku, trannum, stype
df_biloxi = df_biloxi_transact[['sku','trannum','stype']].copy()

#Obtain subset of Biloxi datafraem that contains only purchases
df_biloxi_purchases = df_biloxi[df_biloxi['stype'].str.contains('P')]

#Remove stype column now that all transactions are purchases
del(df_biloxi_purchases['stype'])

#Strip unnecessary spaces and turn data to integers
df_biloxi_purchases['sku'] = df_biloxi_purchases['sku'].str.strip()
df_biloxi_purchases['trannum'] = df_biloxi_purchases['trannum'].str.strip()


df_biloxi_purchases['sku'] = df_biloxi_purchases['sku'].astype(int)
df_biloxi_purchases['trannum'] = df_biloxi_purchases['trannum'].astype(int)


# Create new table in database, biloxi_purchases, and insert data into the new table
cur.execute("CREATE TABLE jaa977_schema.biloxi_purchases (sku integer PRIMARY KEY, trannum integer);")
engine=create_engine('postgresql+psycopg2://jaa977:jaa977_pw@gallery.iems.northwestern.edu:5432/iems308')
df_biloxi_purchases.to_sql('biloxi_purchases',engine,schema = 'jaa977_schema')
connection.commit()

#Close the connection
cur.close()
connection.close()
#%% FOR USE AFTER NEW TABLE CREATED ONLY - BEGIN ASSOCIATION RULE ANALYSIS

import numpy as np
import pandas as pd
import psycopg2
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#Connect to database
connection = psycopg2.connect(host = "gallery.iems.northwestern.edu", user = 'jaa977', password = 'jaa977_pw', dbname = 'iems308')

#Establish cursor
cur = connection.cursor()

#Obtain dataframe from biloxi_transactions
cur.execute("SELECT sku, trannum FROM jaa977_schema.biloxi_purchases")
df_biloxi_purchases = pd.DataFrame(cur.fetchall())
df_biloxi_purchases.columns = ['sku','trannum']

#Close the connection
cur.close()
connection.close()

#Add column to df_biloxi_purchases that specifies number of times SKU appears
sku_counts_series = df_biloxi_purchases['sku'].value_counts()
df_sku_counts = sku_counts_series.to_frame()
df_sku_counts = df_sku_counts.reset_index()
df_sku_counts.columns = ['sku','count']
df_biloxi_with_counts = pd.merge(df_biloxi_purchases,df_sku_counts, on=['sku'])

#Remove duplicates where both sku and trannum match other records
df_biloxi_removed_duplicates = df_biloxi_with_counts.drop_duplicates()

#Create subset of df_biloxi_removed_duplicates where all counts are 5 or higher
df_biloxi_before_encoding = df_biloxi_removed_duplicates[df_biloxi_removed_duplicates['count'] > 4]

#One hot encode the transactions so that each row is a unique transaction and each column denotes 
#the presence of an SKU (1) or the lack of an SKU (0) for the given transaction
df_biloxi_encoded = pd.get_dummies(df_biloxi_before_encoding,columns=['sku'])
del(df_biloxi_encoded['count'])
df_biloxi_reencoded = df_biloxi_encoded.groupby('trannum').sum()

#Run the apriori algorithm
frequent_itemsets = apriori(df_biloxi_reencoded,min_support=0.103, use_colnames=True)
rules = association_rules(frequent_itemsets,metric="lift")

#Eliminate rules with antecedents having over 2 items
good_rules = rules[rules['antecedants'].map(len)<=2]

#Obtain a subset of the association rules where support, confidence, and lift all meet minimum threshold values
#or if any one of support, confidence, or lift meet a higher threshold value
good_rules2 = good_rules[((good_rules['support']>.25) & (good_rules['confidence']>.5) & (good_rules['lift']>3)) | \
             ((good_rules['support']>.6) | (good_rules['confidence']>.75) | (good_rules['lift']>4))]
good_rules2 = good_rules2.reset_index()

#Determine the total number of unique antecedent SKUs in good_rules2
antecedents = []
for i in range(0,np.shape(good_rules2)[0]):
    antecedents.append(list(good_rules2['antecedants'].iloc[i])[0])
    if len(good_rules2['antecedants'].iloc[i])==2:
        antecedents.append(list(good_rules2['antecedants'].iloc[i])[1])
antecedents = np.asarray(antecedents)

#Add most frequently bought SKUs to the list (provided they aren't already in the list)
frequent_counts = df_biloxi_before_encoding.sort_values(by='count',ascending=False).drop_duplicates(subset='sku')

antecedent_integers = []
for num in antecedents:
    antecedent_integers.append(int(num[4:]))

counter = 0    
total_skus = np.unique(antecedent_integers)
while len(total_skus)<100:
    if frequent_counts['sku'].iloc[counter] not in total_skus:
        total_skus = np.append(total_skus,frequent_counts['sku'].iloc[counter])
    counter = counter + 1

#Output final SKUs to move
print(total_skus)