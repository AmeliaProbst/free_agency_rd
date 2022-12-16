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
#os.chdir('C:\\Users\\Richard Clark\\OneDrive\\Documents\\PFF\\git\\mock_free_agency_rd\\code\\contract_offer')

contract_year = 2022
player_id = 11789 #Cam Robinson

#Ts = pd.read_csv('T_data.csv')
players = pd.read_csv('player_data.csv')
Ts = players[players['position'] == 'T']
contracts = pd.read_csv('contract_data.csv')

TContracts = contracts[contracts['Position'] == 'T']

TContracts = TContracts.drop(columns=['player_id'])
TContracts = TContracts.rename(columns={'pff_id':'player_id'})
TContracts = TContracts[TContracts['player_id'].notna()]
TContracts['player_id'] = TContracts['player_id'].astype('int64')

T_y1 = Ts.copy()
T_y2 = Ts.copy()

T_y1 = T_y1.add_suffix('_y1')
T_y2 = T_y2.add_suffix('_y2')

T_y1 = T_y1.rename(columns={'player_id_y1': 'player_id', 'season_y1': 'contract_year_1'})
T_y2 = T_y2.rename(columns={'player_id_y2': 'player_id', 'season_y2': 'contract_year_2'})

TContracts = pd.merge(TContracts, T_y1, on=['player_id', 'contract_year_1'], how='left')
TContracts = pd.merge(TContracts, T_y2, on=['player_id', 'contract_year_2'], how='left')

#TContracts = pd.read_csv('T_contract_data.csv')
TContracts['nameAndContractYear'] = TContracts['Name'] + '_' + TContracts['year_signed'].astype('str')
TContracts = TContracts.drop_duplicates(subset=['nameAndContractYear'])
TContracts['grade'] = TContracts[['offense_y1', 'offense_y2']].mean(axis=1)
TContracts['WAR'] = TContracts[['war_y1', 'war_y2']].mean(axis=1)
TContracts['pass_block_grade'] = TContracts[['pass_block_y1', 'pass_block_y2']].mean(axis=1)
TContracts['run_block_grade'] = TContracts[['run_block_y1', 'run_block_y2']].mean(axis=1)
TContracts['pressure_allowed'] = TContracts[['pressure_allowed_y1', 'pressure_allowed_y2']].sum(axis=1)
TContracts['hurry_allowed'] = TContracts[['hurry_allowed_y1', 'hurry_allowed_y2']].sum(axis=1)
TContracts['hit_allowed'] = TContracts[['hit_allowed_y1', 'hit_allowed_y2']].sum(axis=1)
TContracts['sack_allowed'] = TContracts[['sack_allowed_y1', 'sack_allowed_y2']].sum(axis=1)
TContracts['WAR_Percentile'] = TContracts['WAR'].rank(pct=True)
TContracts['draft_round'] = TContracts['Draft Round_y1'].fillna(0).astype('int')


Ts = Ts[Ts['player_id'] == player_id]

#WRs['year_signed'] = contract_year
#WRs['contract_year_1'] = WRs['year_signed'] - 1
#WRs['contract_year_2'] = WRs['year_signed'] - 2

T_y1 = Ts[Ts['season'] == contract_year - 1]
T_y2 = Ts[Ts['season'] == contract_year - 2]

T_y1 = T_y1.add_suffix('_y1')
T_y2 = T_y2.add_suffix('_y2')

T_y1 = T_y1.rename(columns={'player_id_y1': 'player_id', 'season_y1': 'contract_year_1'})
T_y2 = T_y2.rename(columns={'player_id_y2': 'player_id', 'season_y2': 'contract_year_2'})

#WRs['year_signed'] = contract_year
Ts['contract_year_1'] = contract_year - 1
Ts['contract_year_2'] = contract_year - 2
Ts = Ts[['player_id', 'contract_year_1', 'contract_year_2']]
Ts = Ts.drop_duplicates()

player_info = pd.merge(Ts, T_y1, on=['player_id', 'contract_year_1'], how='left')
player_info = pd.merge(player_info, T_y2, on=['player_id', 'contract_year_2'], how='left')
player_info['grade'] = player_info[['offense_y1', 'offense_y2']].mean(axis=1)
player_info['WAR'] = player_info[['war_y1', 'war_y2']].mean(axis=1)
player_info['pass_block_grade'] = player_info[['pass_block_y1', 'pass_block_y2']].mean(axis=1)
player_info['run_block_grade'] = player_info[['run_block_y1', 'run_block_y2']].mean(axis=1)
player_info['pressure_allowed'] = player_info[['pressure_allowed_y1', 'pressure_allowed_y2']].sum(axis=1)
player_info['hurry_allowed'] = player_info[['hurry_allowed_y1', 'hurry_allowed_y2']].sum(axis=1)
player_info['hit_allowed'] = player_info[['hit_allowed_y1', 'hit_allowed_y2']].sum(axis=1)
player_info['sack_allowed'] = player_info[['sack_allowed_y1', 'sack_allowed_y2']].sum(axis=1)
player_info['WAR_Percentile'] = stats.percentileofscore(TContracts['WAR'], player_info['WAR'][0])/100
player_info['draft_round'] = player_info['Draft Round_y1'].fillna(0).astype('int')


player_grade = player_info['grade'].values[player_info['player_id'] == player_id][0]

offense_cluster_cols = [
    'grade',
    'WAR',
    'pressure_allowed',
    'hurry_allowed',
    'hit_allowed',
    'sack_allowed',
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
    'RT_total_snaps_y1',
    'LT_total_snaps_y1',
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
    'RT_total_snaps_y2',
    'LT_total_snaps_y2'
]

X = TContracts[offense_cluster_cols].fillna(0)
kmeanModel = KMeans(n_clusters=4)#.fit(X)
kmeanModel.fit(X)
TContracts['cluster'] = kmeanModel.labels_

player_cluster = kmeanModel.predict(player_info[offense_cluster_cols])

TContracts['playerComp'] = TContracts['grade'].sub(player_grade).abs()
top_10_similar = TContracts[TContracts['cluster'] == player_cluster[0]].nsmallest(10, 'playerComp') #should be top 10 similar players plus player to compare
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
#TODO try quantile regression
offense_regression_cols = [
    #'grade',
    #'WAR',
    'WAR_Percentile',
    #'pass_block_grade',
    #'run_block_grade',
    'pressure_allowed',
    'hurry_allowed',
    'hit_allowed',
    'sack_allowed',
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

#similar_players = TContracts[TContracts['cluster'] == player_cluster[0]]
#similar_players = similar_players.dropna(subset=offense_cluster_cols)
#X = similar_players[offense_cluster_cols]
#y = similar_players[y_col]

TContracts = TContracts.dropna(subset=offense_regression_cols)
X = TContracts[offense_regression_cols]
y = TContracts[y_col]

free_agent = player_info[offense_regression_cols]

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
poly_reg = PolynomialFeatures(degree=2)
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
print(model_100.predict(free_agent)[0]*1.07)#*208200000)
#print('linear regression model...')
#print(model.predict(free_agent)[0]*208200000)
print('R2 score')
print(model_50.score(X, y))

'''
def viz_polymonial():
    plt.scatter(X, y, color='red')
    plt.plot(X, model.predict(poly_reg.fit_transform(X)), color='blue')
    plt.title('Truth or Bluff (Linear Regression)')
    plt.xlabel('Grade')
    plt.ylabel('APY%')
    plt.show()
    return
viz_polymonial()
'''


#try if WAR in within range of top 20 Grade contracts, then use 10 top APY% contracts most similar to build contract
#look into tangental shaped curve