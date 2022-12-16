import pandas as pd
import numpy as np
import os
import requests
import re
import s3fs
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from dotenv import load_dotenv
from scipy import stats

#ensure correct file path
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

contract_year = 2022
#player_id = 38334 #Kyler Murray
#player_id = 46416 #Lamar Jackson
player_id = 7077 #Russell Wilson


#QBs = pd.read_csv('QB_data.csv')
players = pd.read_csv('player_data.csv')
QBs = players[players['position'] == 'QB']
contracts = pd.read_csv('contract_data.csv')

QBContracts = contracts[contracts['Position'] == 'QB']

#QBContracts = pd.merge(QBContracts, pff_ids[['player_id', 'pff_id']], on='player_id', how='left')

QBContracts = QBContracts.drop(columns=['player_id'])
QBContracts = QBContracts.rename(columns={'pff_id':'player_id'})
QBContracts = QBContracts[QBContracts['player_id'].notna()]
QBContracts['player_id'] = QBContracts['player_id'].astype('int64')

QBs_y1 = QBs.copy()
QBs_y2 = QBs.copy()

QBs_y1 = QBs_y1.add_suffix('_y1')
QBs_y2 = QBs_y2.add_suffix('_y2')

QBs_y1 = QBs_y1.rename(columns={'player_id_y1': 'player_id', 'season_y1': 'contract_year_1'})
QBs_y2 = QBs_y2.rename(columns={'player_id_y2': 'player_id', 'season_y2': 'contract_year_2'})

QBContracts = pd.merge(QBContracts, QBs_y1, on=['player_id', 'contract_year_1'], how='left')
QBContracts = pd.merge(QBContracts, QBs_y2, on=['player_id', 'contract_year_2'], how='left')

#QBContracts = pd.read_csv('QB_contract_data.csv')
QBContracts['nameAndContractYear'] = QBContracts['Name'] + '_' + QBContracts['year_signed'].astype('str')
QBContracts = QBContracts.drop_duplicates(subset=['nameAndContractYear'])
QBContracts['grade'] = QBContracts[['offense_y1', 'offense_y2']].mean(axis=1)
QBContracts['WAR'] = QBContracts[['war_y1', 'war_y2']].mean(axis=1)
#QBContracts['receiving_grade'] = QBContracts[['receiving_y1', 'receiving_y2']].mean(axis=1)
#QBContracts['run_grade'] = QBContracts[['run_y1', 'run_y2']].mean(axis=1)
QBContracts['pass_grade'] = QBContracts[['pass_y1', 'pass_y2']].mean(axis=1)
QBContracts['pass_grade'] = QBContracts[['pass_y1', 'pass_y2']].mean(axis=1)
QBContracts['pass_yards'] = QBContracts[['pass_yards_y1', 'pass_yards_y2']].sum(axis=1)
QBContracts['pass_completions'] = QBContracts[['pass_completion_y1', 'pass_completion_y2']].sum(axis=1)
QBContracts['pass_attempts'] = QBContracts[['pass_attempt_y1', 'pass_attempt_y2']].sum(axis=1)
QBContracts['pass_touchdowns'] = QBContracts[['pass_touchdown_y1', 'pass_touchdown_y2']].sum(axis=1)
QBContracts['interceptions'] = QBContracts[['interception_y1', 'interception_y2']].sum(axis=1)
QBContracts['WAR_Percentile'] = QBContracts['WAR'].rank(pct=True)
QBContracts['draft_round'] = QBContracts['Draft Round_y1'].fillna(0).astype('int')

QBs = QBs[QBs['player_id'] == player_id]

#QBs['year_signed'] = contract_year
#QBs['contract_year_1'] = QBs['year_signed'] - 1
#QBs['contract_year_2'] = QBs['year_signed'] - 2

QBs_y1 = QBs[QBs['season'] == contract_year - 1]
QBs_y2 = QBs[QBs['season'] == contract_year - 2]

QBs_y1 = QBs_y1.add_suffix('_y1')
QBs_y2 = QBs_y2.add_suffix('_y2')

QBs_y1 = QBs_y1.rename(columns={'player_id_y1': 'player_id', 'season_y1': 'contract_year_1'})
QBs_y2 = QBs_y2.rename(columns={'player_id_y2': 'player_id', 'season_y2': 'contract_year_2'})

#QBs['year_signed'] = contract_year
QBs['contract_year_1'] = contract_year - 1
QBs['contract_year_2'] = contract_year - 2
QBs = QBs[['player_id', 'contract_year_1', 'contract_year_2']]
QBs = QBs.drop_duplicates()

player_info = pd.merge(QBs, QBs_y1, on=['player_id', 'contract_year_1'], how='left')
player_info = pd.merge(player_info, QBs_y2, on=['player_id', 'contract_year_2'], how='left')
player_info['grade'] = player_info[['offense_y1', 'offense_y2']].mean(axis=1)
player_info['WAR'] = player_info[['war_y1', 'war_y2']].mean(axis=1)
#player_info['receiving_grade'] = player_info[['receiving_y1', 'receiving_y2']].mean(axis=1)
#player_info['run_grade'] = player_info[['run_y1', 'run_y2']].mean(axis=1)
player_info['pass_grade'] = player_info[['pass_y1', 'pass_y2']].mean(axis=1)
player_info['pass_yards'] = player_info[['pass_yards_y1', 'pass_yards_y2']].sum(axis=1)
player_info['pass_completions'] = player_info[['pass_completion_y1', 'pass_completion_y2']].sum(axis=1)
player_info['pass_attempts'] = player_info[['pass_attempt_y1', 'pass_attempt_y2']].sum(axis=1)
player_info['pass_touchdowns'] = player_info[['pass_touchdown_y1', 'pass_touchdown_y2']].sum(axis=1)
player_info['interceptions'] = player_info[['interception_y1', 'interception_y2']].sum(axis=1)
player_info['WAR_Percentile'] = stats.percentileofscore(QBContracts['WAR'], player_info['WAR'][0])/100
player_info['draft_round'] = player_info['Draft Round_y1'].fillna(0).astype('int')

player_grade = player_info['grade'].values[player_info['player_id'] == player_id][0]

offense_cluster_cols = [
    'grade',
    'WAR',
    'pass_yards',
    'pass_completions',
    'pass_attempts',
    'pass_touchdowns',
    'interceptions',
    'offense_y1',
    #'pass_block_y1',
    #'run_block_y1',
    #'receiving_y1',
    #'run_y1',
    'war_y1',
    #'waa_y1',
    #'height_y1',
    #'weight_y1',
    'year_in_league_y1',
    'total_alignments_played_y1',
    'total_positions_played_y1',
    'total_snap_count_y1',
    'weeks_played_y1',
    'snap_count_per_week_played_y1',
    'QB_total_snaps_y1',
    #'SQB_total_snaps_y1',
    'offense_y2',
    #'pass_block_y2',
    #'run_block_y2',
    #'receiving_y2',
    #'run_y2',
    'war_y2',
    #'waa_y2',
    #'height_y2',
    #'weight_y2',
    'year_in_league_y2',
    'total_alignments_played_y2',
    'total_positions_played_y2',
    'total_snap_count_y2',
    'weeks_played_y2',
    'snap_count_per_week_played_y2',
    'QB_total_snaps_y2',
    #'SQB_total_snaps_y2'
]

X = QBContracts[offense_cluster_cols].fillna(0)
kmeanModel = KMeans(n_clusters=5)#.fit(X)
kmeanModel.fit(X)
QBContracts['cluster'] = kmeanModel.labels_

player_cluster = kmeanModel.predict(player_info[offense_cluster_cols])


QBContracts['playerComp'] = QBContracts['grade'].sub(player_grade).abs()
top_10_similar = QBContracts[QBContracts['cluster'] == player_cluster[0]].nsmallest(10, 'playerComp') #should be top 10 similar players plus player to compare
top_10_similar = top_10_similar[top_10_similar['player_id'] != player_id]

print('most similar players and contract years...')
print(top_10_similar['nameAndContractYear'].head(10))
'''
print('25th percentile contract based on similar players...')
print(top_10_similar['APY%'].quantile(.25)*208200000)
print('50th percentile contract based on similar players...')
print(top_10_similar['APY%'].quantile(.5)*208200000)
print('75th percentile contract based on similar players...')
print(top_10_similar['APY%'].quantile(.75)*208200000)
print('99th percentile contract based on similar players...')
print(top_10_similar['APY%'].quantile(.99)*208200000)
print('100th percentile contract with cap increase based on similar players...')
print(top_10_similar['APY%'].quantile(1)*208200000*1.07)
'''


#TODO increase sample size. take players from last 5-10 years. 
# take two seasons leading up to new contract for player stats and new contract APY (single player could have multiple entries)


offense_regression_cols = [
    #'grade',
    #'WAR',
    'WAR_Percentile',
    #'run_grade',
    #'receiving_grade',
    #'pass_grade',
    'pass_yards',
    'pass_completions',
    'pass_attempts',
    'pass_touchdowns',
    'interceptions',
    #'offense_y1',
    #'pass_block_y1',
    #'run_block_y1',
    #'receiving_y1',
    #'run_y1',
    #'war_y1',
    #'waa_y1',
    #'height_y1',
    #'weight_y1',
    #'year_in_league_y1',
    #'total_alignments_played_y1',
    #'total_positions_played_y1',
    #'total_snap_count_y1',
    #'weeks_played_y1',
    #'snap_count_per_week_played_y1',
    #'QB_total_snaps_y1',
    #'SQB_total_snaps_y1',
    #'offense_y2',
    #'pass_block_y2',
    #'run_block_y2',
    #'receiving_y2',
    #'run_y2',
    #'war_y2',
    #'waa_y2',
    #'height_y2',
    #'weight_y2',
    #'year_in_league_y2',
    #'total_alignments_played_y2',
    #'total_positions_played_y2',
    #'total_snap_count_y2',
    #'weeks_played_y2',
    #'snap_count_per_week_played_y2',
    #'QB_total_snaps_y2',
    #'SQB_total_snaps_y2'
    'draft_round'
]
#y_col = 'APY%'
y_col = 'APY_Adj'

similar_players = QBContracts[QBContracts['cluster'] == player_cluster[0]]
similar_players = similar_players.dropna(subset=offense_regression_cols)
X = similar_players[offense_regression_cols]
y = similar_players[y_col]

free_agent = player_info[offense_regression_cols]

if (free_agent['WAR_Percentile'] > .9).item() & (free_agent['draft_round'] > 3).item():
    X = X.drop(columns=['draft_round'])
    free_agent = free_agent.drop(columns=['draft_round'])
    #free_agent['draft_round'] = 1
#elif (free_agent['WAR_Percentile'] > .94).item() & (free_agent['draft_round'] > 3).item():
#    free_agent['draft_round'] = 2
#elif (free_agent['WAR_Percentile'] > .9).item() & (free_agent['draft_round'] > 3).item():
#    free_agent['draft_round'] = 3

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=1)
X = poly_reg.fit_transform(X)
free_agent = poly_reg.fit_transform(free_agent)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model = lr.fit(X, y)

from sklearn.ensemble import GradientBoostingRegressor
#build models for low middle and high
gbr_25 = GradientBoostingRegressor(loss='quantile', alpha=0.25)
gbr_50 = GradientBoostingRegressor(loss='quantile', alpha=0.5)
gbr_75 = GradientBoostingRegressor(loss='quantile', alpha=0.8)
gbr_99 = GradientBoostingRegressor(loss='quantile', alpha=0.99)
gbr_100 = GradientBoostingRegressor(loss='quantile', alpha=0.9999)

model_25 = gbr_25.fit(X, y)
model_50 = gbr_50.fit(X, y)
model_75 = gbr_75.fit(X, y)
model_99 = gbr_99.fit(X, y)
model_100 = gbr_100.fit(X, y)

'''print('25th percentile contract based on similar players...')
print(model_25.predict(free_agent)[0]*208200000)
print('50th percentile contract based on similar players...')
print(model_50.predict(free_agent)[0]*208200000)
print('75th percentile contract based on similar players...')
print(model_75.predict(free_agent)[0]*208200000)
print('99th percentile contract based on similar players...')
print(model_99.predict(free_agent)[0]*208200000)
print('100th percentile contract based on similar players...')
print(model_100.predict(free_agent)[0]*208200000*1.07)
#print('linear regression model...')
#print(model.predict(free_agent)[0]*208200000)
print('R2 score')
print(model_50.score(X, y))'''

'''print('25th percentile contract based on similar players...')
print(model_25.predict(free_agent)[0])#*208200000)
print('50th percentile contract based on similar players...')
print(model_50.predict(free_agent)[0])#*208200000)
print('75th percentile contract based on similar players...')
print(model_75.predict(free_agent)[0])#*208200000)
print('99th percentile contract based on similar players...')
print(model_99.predict(free_agent)[0])#*208200000)
print('100th percentile contract based on similar players...')
print(model_100.predict(free_agent)[0])#*1.07)#*208200000)
#print('linear regression model...')
#print(model.predict(free_agent)[0])#*208200000)
print('R2 score')
print(model_50.score(X, y))'''

print('25th percentile contract based on similar players...')
print(model.predict(free_agent)[0]*.75)#*208200000)
print('50th percentile contract based on similar players...')
print(model.predict(free_agent)[0])#*208200000)
print('75th percentile contract based on similar players...')
print(model.predict(free_agent)[0]*1.25)#*208200000)
print('99th percentile contract based on similar players...')
print(model.predict(free_agent)[0]*1.5)#*208200000)
print('100th percentile contract based on similar players...')
print(model.predict(free_agent)[0]*1.57)#*1.07)#*208200000)
#print('linear regression model...')
#print(model.predict(free_agent)[0])#*208200000)
print('R2 score')
print(model.score(X, y))

#finish all contract offer models for every position
#refine all contract offer models with more positional data (i.e. receiving yards, rushing yards, etc.)
#have a strong decision model for at least one position finished

# look into why years spike in contract
# i.e. a lot of high war players entering FA in the next year
# premium vs non premium position
# take a look at max median and mean WAR

# for rookies and 1-2 rounders look at two best years
# for players leaving rookie deals who were 1-2 rounders, look at 2 best years