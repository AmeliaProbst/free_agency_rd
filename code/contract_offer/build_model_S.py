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
player_id = 51368 #Derwin James

players = pd.read_csv('player_data.csv')
Safeties = players[players['position'] == 'S']
#Safeties = pd.read_csv('safety_data.csv')
contracts = pd.read_csv('contract_data.csv')


SContracts = contracts[contracts['Position'] == 'S']

SContracts = SContracts.drop(columns=['player_id'])
SContracts = SContracts.rename(columns={'pff_id':'player_id'})
SContracts = SContracts[SContracts['player_id'].notna()]
SContracts['player_id'] = SContracts['player_id'].astype('int64')

S_y1 = Safeties.copy()
S_y2 = Safeties.copy()

S_y1 = S_y1.add_suffix('_y1')
S_y2 = S_y2.add_suffix('_y2')

S_y1 = S_y1.rename(columns={'player_id_y1': 'player_id', 'season_y1': 'contract_year_1'})
S_y2 = S_y2.rename(columns={'player_id_y2': 'player_id', 'season_y2': 'contract_year_2'})

SContracts = pd.merge(SContracts, S_y1, on=['player_id', 'contract_year_1'], how='left')
SContracts = pd.merge(SContracts, S_y2, on=['player_id', 'contract_year_2'], how='left')

#SContracts = pd.read_csv('safety_contract_data.csv')
SContracts['nameAndContractYear'] = SContracts['Name'] + '_' + SContracts['year_signed'].astype('str')
SContracts = SContracts.drop_duplicates(subset=['nameAndContractYear'])
SContracts['grade'] = SContracts[['defense_y1', 'defense_y2']].mean(axis=1)
SContracts['WAR'] = SContracts[['war_y1', 'war_y2']].mean(axis=1)
SContracts['hurries'] = SContracts[['hurries_y1', 'hurries_y2']].sum(axis=1)
SContracts['pressures'] = SContracts[['pressures_y1', 'pressures_y2']].sum(axis=1)
SContracts['sacks'] = SContracts[['sacks_y1', 'sacks_y2']].sum(axis=1)
SContracts['forced_fumbles'] = SContracts[['forced_fumble_y1', 'forced_fumble_y2']].sum(axis=1)
SContracts['stops'] = SContracts[['stops_y1', 'stops_y2']].sum(axis=1)
SContracts['tackles'] = SContracts[['tackle_y1', 'tackle_y2']].sum(axis=1)
SContracts['assisted_tackles'] = SContracts[['assisted_tackle_y1', 'assisted_tackle_y2']].sum(axis=1)
SContracts['interceptions'] = SContracts[['interceptions_y1', 'interceptions_y2']].sum(axis=1)
SContracts['missed_tackles'] = SContracts[['missed_tackle_y1', 'missed_tackle_y2']].sum(axis=1)
SContracts['pass_breakups'] = SContracts[['pass_breakups_y1', 'pass_breakups_y2']].sum(axis=1)
SContracts['penalties'] = SContracts[['penalty_y1', 'penalty_y2']].sum(axis=1)
SContracts['fumble_recoveries'] = SContracts[['fumble_recovery_y1', 'fumble_recovery_y2']].sum(axis=1)
SContracts['hits'] = SContracts[['hits_y1', 'hits_y2']].sum(axis=1)
SContracts['batted_passes'] = SContracts[['batted_pass_y1', 'batted_pass_y2']].sum(axis=1)
SContracts['WAR_Percentile'] = SContracts['WAR'].rank(pct=True)
SContracts['draft_round'] = SContracts['Draft Round_y1'].fillna(0).astype('int')

Safeties = Safeties[Safeties['player_id'] == player_id]

#WRs['year_signed'] = contract_year
#WRs['contract_year_1'] = WRs['year_signed'] - 1
#WRs['contract_year_2'] = WRs['year_signed'] - 2

S_y1 = Safeties[Safeties['season'] == contract_year - 3]
S_y2 = Safeties[Safeties['season'] == contract_year - 4]

S_y1 = S_y1.add_suffix('_y1')
S_y2 = S_y2.add_suffix('_y2')

S_y1 = S_y1.rename(columns={'player_id_y1': 'player_id', 'season_y1': 'contract_year_1'})
S_y2 = S_y2.rename(columns={'player_id_y2': 'player_id', 'season_y2': 'contract_year_2'})

#WRs['year_signed'] = contract_year
Safeties['contract_year_1'] = contract_year - 3
Safeties['contract_year_2'] = contract_year - 4
Safeties = Safeties[['player_id', 'contract_year_1', 'contract_year_2']]
Safeties = Safeties.drop_duplicates()

player_info = pd.merge(Safeties, S_y1, on=['player_id', 'contract_year_1'], how='left')
player_info = pd.merge(player_info, S_y2, on=['player_id', 'contract_year_2'], how='left')
player_info['grade'] = player_info[['defense_y1', 'defense_y2']].mean(axis=1)
player_info['WAR'] = player_info[['war_y1', 'war_y2']].mean(axis=1)
player_info['hurries'] = player_info[['hurries_y1', 'hurries_y2']].sum(axis=1)
player_info['pressures'] = player_info[['pressures_y1', 'pressures_y2']].sum(axis=1)
player_info['sacks'] = player_info[['sacks_y1', 'sacks_y2']].sum(axis=1)
player_info['forced_fumbles'] = player_info[['forced_fumble_y1', 'forced_fumble_y2']].sum(axis=1)
player_info['stops'] = player_info[['stops_y1', 'stops_y2']].sum(axis=1)
player_info['tackles'] = player_info[['tackle_y1', 'tackle_y2']].sum(axis=1)
player_info['assisted_tackles'] = player_info[['assisted_tackle_y1', 'assisted_tackle_y2']].sum(axis=1)
player_info['interceptions'] = player_info[['interceptions_y1', 'interceptions_y2']].sum(axis=1)
player_info['missed_tackles'] = player_info[['missed_tackle_y1', 'missed_tackle_y2']].sum(axis=1)
player_info['pass_breakups'] = player_info[['pass_breakups_y1', 'pass_breakups_y2']].sum(axis=1)
player_info['penalties'] = player_info[['penalty_y1', 'penalty_y2']].sum(axis=1)
player_info['fumble_recoveries'] = player_info[['fumble_recovery_y1', 'fumble_recovery_y2']].sum(axis=1)
player_info['hits'] = player_info[['hits_y1', 'hits_y2']].sum(axis=1)
player_info['batted_passes'] = player_info[['batted_pass_y1', 'batted_pass_y2']].sum(axis=1)
player_info['WAR_Percentile'] = stats.percentileofscore(SContracts['WAR'], player_info['WAR'][0])/100
player_info['draft_round'] = player_info['Draft Round_y1'].fillna(0).astype('int')

player_grade = player_info['grade'].values[player_info['player_id'] == player_id][0]

defense_cluster_cols = [
    'grade',
    'WAR',
    'hurries',
    'pressures',
    'sacks',
    'forced_fumbles',
    'stops',
    'tackles',
    'assisted_tackles',
    'interceptions',
    'missed_tackles',
    'pass_breakups',
    'penalties',
    'fumble_recoveries',
    'hits',
    'batted_passes',
    'defense_y1',
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
    'SS_total_snaps_y1',
    'FS_total_snaps_y1',
    'defense_y2',
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
    'SS_total_snaps_y2',
    'FS_total_snaps_y2'
]

X = SContracts[defense_cluster_cols].fillna(0)
kmeanModel = KMeans(n_clusters=4)#.fit(X)
kmeanModel.fit(X)
SContracts['cluster'] = kmeanModel.labels_

player_cluster = kmeanModel.predict(player_info[defense_cluster_cols])


SContracts['playerComp'] = SContracts['grade'].sub(player_grade).abs()
top_10_similar = SContracts[SContracts['cluster'] == player_cluster[0]].nsmallest(10, 'playerComp') #should be top 10 similar players plus player to compare
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

offense_cluster_cols = [
    #'grade',
    #'WAR',
    'WAR_Percentile',
    'hurries',
    'pressures',
    'sacks',
    'forced_fumbles',
    'stops',
    'tackles',
    'assisted_tackles',
    'interceptions',
    'missed_tackles',
    'pass_breakups',
    'penalties',
    'fumble_recoveries',
    'hits',
    'batted_passes',
    #'pass_block_grade',
    #'run_block_grade',
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
    #'RG_total_snaps_y1',
    #'LG_total_snaps_y1',
    #'C_total_snaps_y1',
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
    #'RG_total_snaps_y2',
    #'LG_total_snaps_y2',
    #'C_total_snaps_y1'
    'draft_round'
]
#y_col = 'APY%'
y_col = 'APY_Adj'

#similar_players = IOLContracts[IOLContracts['cluster'] == player_cluster[0]]
#similar_players = similar_players.dropna(subset=offense_cluster_cols)
#X = similar_players[offense_cluster_cols]
#y = similar_players[y_col]

SContracts = SContracts.dropna(subset=offense_cluster_cols)
X = SContracts[offense_cluster_cols]
y = SContracts[y_col]

free_agent = player_info[offense_cluster_cols]

if (free_agent['WAR_Percentile'] > .9).item() & (free_agent['draft_round'] > 3).item():
    X = X.drop(columns=['draft_round'])
    free_agent = free_agent.drop(columns=['draft_round'])
    #free_agent['draft_round'] = 1
#elif (free_agent['WAR_Percentile'] > .94).item() & (free_agent['draft_round'] > 3).item():
#    free_agent['draft_round'] = 2
#elif (free_agent['WAR_Percentile'] > .9).item() & (free_agent['draft_round'] > 3).item():
#    free_agent['draft_round'] = 3

'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
free_agent = scaler.fit_transform(free_agent)
'''

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
X = poly_reg.fit_transform(X)
free_agent = poly_reg.fit_transform(free_agent)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model = lr.fit(X, y)

from sklearn.ensemble import GradientBoostingRegressor
#build models for low middle and high
gbr_25 = GradientBoostingRegressor(loss='quantile', alpha=0.25)
gbr_50 = GradientBoostingRegressor(loss='quantile', alpha=0.5)
gbr_75 = GradientBoostingRegressor(loss='quantile', alpha=0.75)
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

print('25th percentile contract based on similar players...')
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
print(model_50.score(X, y))

'''print('25th percentile contract based on similar players...')
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
print(model.score(X, y))'''

#TODO increase sample size. take players from last 5-10 years. 
# take two seasons leading up to new contract for player stats and new contract APY (single player could have multiple entries)