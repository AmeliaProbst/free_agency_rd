import pandas as pd
import numpy as np
import os
import requests
import re
import s3fs
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from dotenv import load_dotenv

#ensure correct file path
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
#os.chdir('C:\\Users\\Richard Clark\\OneDrive\\Documents\\PFF\\git\\mock_free_agency_rd\\code\\contract_offer')

#### Helper Functions
def pull_api(path):
    load_dotenv()
    # ACQUIRE JSON WEB TOKEN (JWT) WHICH HAS A LIMITED LIFE SPAN
    token_url = os.getenv('PFF_API_URL') + '/auth/login'
    jwt = requests.post(token_url, headers={'x-api-key': os.getenv('PFF_API_KEY')}).json()['jwt']

    # GET DATA FROM API
    x = requests.get(os.getenv('PFF_API_URL') + '/' + path, headers={'Authorization': 'Bearer ' + jwt, 'Accept-Encoding': 'gzip'})
    return pd.json_normalize(pd.DataFrame(x.json()).iloc[:, 0])

def pull_s3(obj_base, season_start=2006, season_end=2020, bucket="api", join_odds=False):
    load_dotenv()
    if not (bucket in ["api", "ml"]):
        raise Exception("Bucket must be either 'api' or 'ml'!")
    else:
        if bucket == "api":
            bucket_full = os.getenv('FEEDS_BUCKET') 
        else:
            bucket_full = os.getenv('ML_BUCKET')
    # FRESH DF
    df_list = []
    i = 1
    # IS THE API SEASON DEPENDENT
    seasonal = re.search("%", obj_base) != None
    # ONLY LOOP IF YOU HAVE A SEASON COMPONENT
    if seasonal:
        seasons = range(season_start, season_end + 1)
        for i in seasons:
            obj_name = obj_base % i
            temp = pd.read_csv(
                "s3://%s/%s" % (bucket_full, obj_name),
                dtype={"Season": "object", "Daynum": "object", "GameID": "object"},
                converters={"Season": lambda x: str(x), "Daynum": lambda x: str(x), "GameID": lambda x: str(x)},
                na_values=[""],
                low_memory=False,
            )
            df_list.append(temp)
            i += 1
        df = pd.concat(df_list)
    else:
        df = pd.read_csv(
            "s3://%s/%s" % (bucket_full, obj_base),
            dtype={"Season": "object", "Daynum": "object", "GameID": "object"},
            converters={"Season": lambda x: str(x), "Daynum": lambda x: str(x), "GameID": lambda x: str(x)},
            na_values=[""],
            low_memory=False,
        )
        if join_odds:
            odds = pd.read_csv(
                "s3://%s/flat_files/betting_odds_history_v2.csv" % os.getenv('ML_BUCKET'),
                dtype={"Season": "object", "Daynum": "object", "GameID": "object"},
                converters={"Season": lambda x: str(x), "Daynum": lambda x: str(x), "GameID": lambda x: str(x)},
                na_values=[""],
                low_memory=False,
            )
            df = join_odds(df, odds)
    return df

def write_s3(folder_name, file_name,data):
    load_dotenv()
    bucket = os.getenv('ML_BUCKET')
    storage_options = dict(anon=False, s3_additional_kwargs=dict(ServerSideEncryption="AES256"))
    s3 = s3fs.S3FileSystem(**storage_options)
    if isinstance(data,pd.DataFrame):
        data.to_csv(f's3://{bucket}/{folder_name}/{file_name}.csv', index=False, storage_options=storage_options)
    else:
        with s3.open(f's3://{bucket}/{folder_name}/{file_name}', 'wb') as f:
            f.write(data)

def concat_api(source, first_season, last_season):
    df_list = []
    for i in range(first_season, last_season+1):
        temp = pull_api(source %i)
        df_list.append(temp)
    df = pd.concat(df_list)
    return df

contract_year = 2022
#player_id = 47546 #Christian Kirk
player_id = 47864 #DK Metcalf
#player_id = 11824 #Cooper Kupp
#player_id = 48236 #Darnell Mooney
#player_id = 33441 #Diontae Johnson
#player_id = 48274 #Deebo Samuel
#player_id = 48327 #AJ Brown
#player_id = 10799 #Tyreek Hill

#WRs = pd.read_csv('wr_data.csv')
players = pd.read_csv('player_data.csv')
WRs = players[players['position'] == 'WR']
contracts = pd.read_csv('contract_data.csv')

'''receiving_fantasy = concat_api('/v1/fantasy/nfl/%i/receiving', 2011, 2021)
#TODO seasons before 2021 go up to week 17, others week 18
receiving_totals_season = receiving_fantasy[receiving_fantasy['week'] < 19]
receiving_totals_season = receiving_totals_season.groupby(['target_player_id', 'season'], as_index=False)[['yards', 'reception', 'target', 'touchdown']].sum()
receiving_totals_season = receiving_totals_season.rename(columns={'yards':'receiving_yards', 'target_player_id':'player_id'})

WRs = pd.merge(WRs, receiving_totals_season, on=['player_id', 'season'], how='left')'''

WRContracts = contracts[contracts['Position'] == 'WR']
#WRContracts = pd.merge(WRContracts, pff_ids[['player_id', 'pff_id']], on='player_id', how='left')

WRContracts = WRContracts.drop(columns=['player_id'])
WRContracts = WRContracts.rename(columns={'pff_id':'player_id'})
WRContracts = WRContracts[WRContracts['player_id'].notna()]
WRContracts['player_id'] = WRContracts['player_id'].astype('int64')

WRs_y1 = WRs.copy()
WRs_y2 = WRs.copy()

WRs_y1 = WRs_y1.add_suffix('_y1')
WRs_y2 = WRs_y2.add_suffix('_y2')

WRs_y1 = WRs_y1.rename(columns={'player_id_y1': 'player_id', 'season_y1': 'contract_year_1'})
WRs_y2 = WRs_y2.rename(columns={'player_id_y2': 'player_id', 'season_y2': 'contract_year_2'})

WRContracts = pd.merge(WRContracts, WRs_y1, on=['player_id', 'contract_year_1'], how='left')
WRContracts = pd.merge(WRContracts, WRs_y2, on=['player_id', 'contract_year_2'], how='left')

#WRContracts = pd.read_csv('wr_contract_data.csv')
WRContracts['nameAndContractYear'] = WRContracts['Name'] + '_' + WRContracts['year_signed'].astype('str')
WRContracts = WRContracts.drop_duplicates(subset=['nameAndContractYear'])
WRContracts['grade'] = WRContracts[['offense_y1', 'offense_y2']].mean(axis=1)
WRContracts['WAR'] = WRContracts[['war_y1', 'war_y2']].mean(axis=1)
WRContracts['WAR_Percentile'] = WRContracts['WAR'].rank(pct=True)
WRContracts['receiving_grade'] = WRContracts[['receiving_y1', 'receiving_y2']].mean(axis=1)
WRContracts['receiving_yards'] = WRContracts[['receiving_yards_y1', 'receiving_yards_y2']].sum(axis=1)
WRContracts['receptions'] = WRContracts[['reception_y1', 'reception_y2']].sum(axis=1)
WRContracts['targets'] = WRContracts[['target_y1', 'target_y2']].sum(axis=1)
WRContracts['receiving_touchdowns'] = WRContracts[['receiving_touchdown_y1', 'receiving_touchdown_y2']].sum(axis=1)
WRContracts['draft_round'] = WRContracts['Draft Round_y1'].fillna(0).astype('int')
#WRContracts['draft_round'] = np.where(WRContracts['draft_round'] == 0, 8, WRContracts['draft_round'])

WRs = WRs[WRs['player_id'] == player_id]

#WRs['year_signed'] = contract_year
#WRs['contract_year_1'] = WRs['year_signed'] - 1
#WRs['contract_year_2'] = WRs['year_signed'] - 2

WRs_y1 = WRs[WRs['season'] == contract_year - 1]
WRs_y2 = WRs[WRs['season'] == contract_year - 2]

WRs_y1 = WRs_y1.add_suffix('_y1')
WRs_y2 = WRs_y2.add_suffix('_y2')

WRs_y1 = WRs_y1.rename(columns={'player_id_y1': 'player_id', 'player_y1': 'player', 'season_y1': 'contract_year_1'})
WRs_y2 = WRs_y2.rename(columns={'player_id_y2': 'player_id', 'player_y2': 'player', 'season_y2': 'contract_year_2'})

#WRs['year_signed'] = contract_year
WRs['contract_year_1'] = contract_year - 1
WRs['contract_year_2'] = contract_year - 2
WRs = WRs[['player_id', 'player', 'contract_year_1', 'contract_year_2']]
WRs = WRs.drop_duplicates()

player_info = pd.merge(WRs, WRs_y1, on=['player_id', 'contract_year_1'], how='left')
player_info = pd.merge(player_info, WRs_y2, on=['player_id', 'contract_year_2'], how='left')
player_info['grade'] = player_info[['offense_y1', 'offense_y2']].mean(axis=1)
player_info['WAR'] = player_info[['war_y1', 'war_y2']].mean(axis=1)
player_info['WAR_Percentile'] = stats.percentileofscore(WRContracts['WAR'], player_info['WAR'][0])/100
player_info['receiving_grade'] = player_info[['receiving_y1', 'receiving_y2']].mean(axis=1)
player_info['receiving_yards'] = player_info[['receiving_yards_y1', 'receiving_yards_y2']].sum(axis=1)
player_info['receptions'] = player_info[['reception_y1', 'reception_y2']].sum(axis=1)
player_info['targets'] = player_info[['target_y1', 'target_y2']].sum(axis=1)
player_info['receiving_touchdowns'] = player_info[['receiving_touchdown_y1', 'receiving_touchdown_y2']].sum(axis=1)
player_info['draft_round'] = player_info['Draft Round_y1'].fillna(0).astype('int')
#player_info['draft_round'] = np.where(player_info['draft_round'] == 0, 8, player_info['draft_round'])

player_grade = player_info['offense_y1'].values[player_info['player_id'] == player_id][0]

offense_cluster_cols = [
    'grade',
    #'WAR',
    'WAR_Percentile',
    #'receiving_grade',
    'receiving_yards',
    'receptions',
    'targets',
    'receiving_touchdowns',
    #'offense_y1',
    #'pass_block_y1',
    #'run_block_y1',
    #'receiving_y1',
    #'run_y1',
    #'war_y1',
    #'waa_y1',
    #'height_y1',
    #'weight_y1',
    'year_in_league_y1',
    #'total_alignments_played_y1',
    #'total_positions_played_y1',
    #'total_snap_count_y1',
    #'weeks_played_y1',
    'snap_count_per_week_played_y1',
    'WR_total_snaps_y1',
    'SWR_total_snaps_y1',
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
    'snap_count_per_week_played_y2',
    'WR_total_snaps_y2',
    'SWR_total_snaps_y2',
    'height_in_inches_y1',
    'weight_in_pounds_y1',
    'wingspan_in_inches_y1',
    'arm_length_in_inches_y1',
    'right_hand_size_in_inches_y1',
    'left_hand_size_in_inches_y1',
    'fourty_time_in_seconds_y1',
    'twenty_time_in_seconds_y1',
    'ten_time_in_seconds_y1',
    'twenty_shuttle_in_seconds_y1',
    'three_cone_in_seconds_y1',
    'vertical_jump_in_inches_y1',
    'broad_jump_in_inches_y1',
    'bench_press_in_reps_y1'
]

X = WRContracts[offense_cluster_cols].fillna(0)
kmeanModel = KMeans(n_clusters=5)#.fit(X)
kmeanModel.fit(X)
WRContracts['cluster'] = kmeanModel.labels_

player_cluster = kmeanModel.predict(player_info[offense_cluster_cols].fillna(0))


WRContracts['playerComp'] = WRContracts['offense_y1'].sub(player_grade).abs()
top_10_similar = WRContracts[WRContracts['cluster'] == player_cluster[0]].nsmallest(10, 'playerComp') #should be top 10 similar players plus player to compare
top_10_similar = top_10_similar[top_10_similar['player_id'] != player_id]

#print('most similar players and contract years...')
#print(top_10_similar['nameAndContractYear'].head(10))
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
    #'receiving_grade',
    'receiving_yards',
    'receptions',
    'targets',
    'receiving_touchdowns',
    #'offense_y1',
    #'pass_block_y1',
    #'run_block_y1',
    #'receiving_y1',
    #'run_y1',
    #'war_y1',
    #'waa_y1',
    #'height_y1',
    #'weight_y1',
    'year_in_league_y1',
    #'total_alignments_played_y1',
    #'total_positions_played_y1',
    #'total_snap_count_y1',
    #'weeks_played_y1',
    #'snap_count_per_week_played_y1',
    #'WR_total_snaps_y1',
    #'SWR_total_snaps_y1',
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
    #'WR_total_snaps_y2',
    #'SWR_total_snaps_y2'
    #'height_in_inches_y1',
    #'weight_in_pounds_y1',
    #'wingspan_in_inches_y1',
    #'arm_length_in_inches_y1',
    #'right_hand_size_in_inches_y1',
    #'left_hand_size_in_inches_y1',
    #'fourty_time_in_seconds_y1',
    #'twenty_time_in_seconds_y1',
    #'ten_time_in_seconds_y1',
    #'twenty_shuttle_in_seconds_y1',
    #'three_cone_in_seconds_y1',
    #'vertical_jump_in_inches_y1',
    #'broad_jump_in_inches_y1',
    #'bench_press_in_reps_y1'
    'draft_round',
    #'Draft Round_y1'
]
#y_col = 'APY%'
y_col = 'APY_Adj'

'''similar_players = WRContracts[WRContracts['cluster'] == player_cluster[0]]
similar_players = similar_players.dropna(subset=offense_regression_cols)
X = similar_players[offense_regression_cols]
y = similar_players[y_col]'''

#TODO explore pareto distribution/tail

WRContracts = WRContracts.dropna(subset=offense_regression_cols)
X = WRContracts[offense_regression_cols]#.fillna(0)
y = WRContracts[y_col]

free_agent = player_info[offense_regression_cols]#.fillna(5)

if (free_agent['WAR_Percentile'] > .9).item() & (free_agent['draft_round'] > 3).item():
    X = X.drop(columns=['draft_round'])
    free_agent = free_agent.drop(columns=['draft_round'])
    #free_agent['draft_round'] = 1
#elif (free_agent['WAR_Percentile'] > .94).item() & (free_agent['draft_round'] > 3).item():
#    free_agent['draft_round'] = 2
#elif (free_agent['WAR_Percentile'] > .9).item() & (free_agent['draft_round'] > 3).item():
#    free_agent['draft_round'] = 3

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
X = poly_reg.fit_transform(X)
free_agent = poly_reg.fit_transform(free_agent)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model = lr.fit(X, y)

'''from sklearn.ensemble import GradientBoostingRegressor
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
model_100 = gbr_100.fit(X, y)'''

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

print('Player to negotiate with...')
print(WRs['player'].item())
print()

#print('CONTRACT OFFER OPTION 1')
#print(round(model.predict(free_agent)[0]*.75, -5))#*208200000)
print('CONTRACT SUGGESTION 1')
print(round(model.predict(free_agent)[0], -5))#*208200000)
print('CONTRACT SUGGESTION 2')
print(round(model.predict(free_agent)[0]*1.25, -5))#*208200000)
print('CONTRACT SUGGESTION 3')
print(round(model.predict(free_agent)[0]*1.5, -5))#*208200000)
#print('CONTRACT OFFER OPTION 5')
#print(round(model.predict(free_agent)[0]*1.57, -5))#*1.07)#*208200000)
#print('linear regression model...')
#print(model.predict(free_agent)[0])#*208200000)
#print('R2 score')
#print(model.score(X, y))
print()

lower_bound = model.predict(free_agent)[0]*.75
upper_bound = model.predict(free_agent)[0]*1.57

contract_percentile_list = np.linspace(lower_bound, upper_bound, 1000)

print('Enter APY contract offer')
user_contract_offer = int(input())

accept = stats.percentileofscore(contract_percentile_list, user_contract_offer)
original_accept = accept
#decline = (100 - accept)/2
#counter = (100 - accept)/2
#print(accept)
if (accept >= 0) & (accept <= 20):
    accept = 0
    decline = 100
    counter = 0
elif (accept > 20) & (accept <= 40):
    counter = accept/2
    accept = 0
    decline = 100 - counter
elif (accept > 40) & (accept <= 60):
    decline = (100 - accept)/2
    counter = (100 - accept)/2
elif (accept > 60) & (accept <= 80):
    decline = (100 - accept)/4
    counter = 100 - accept - decline
else:
    decline = 0
    counter = 100 - accept

'''print('percent likelihood of accepting offer...')
print(accept)

print('percent likelihood of declining offer...')
print(decline)

print('percent likelihood of countering offer...')
print(counter)'''
print()

import random
choiceList = ['accept', 'decline', 'counter']

print('Player Decision...')
decision = random.choices(choiceList, weights=(accept, decline, counter))
#decision = ['counter']
print(decision)

print()

if decision == ['accept']:
    print('Congrats on signing ' + WRs['player'].item() + ' to your team!')
elif decision == ['decline']:
    print('Sorry, no deal.')
elif decision == ['counter']:
    if (original_accept > 20) & (original_accept <= 40):
        counter_offer = user_contract_offer*1.2
        print(WRs['player'].item() + ' sends the following counter offer...')
        print(round(counter_offer, -5))
    elif (original_accept > 40) & (original_accept <= 60):
        counter_offer = user_contract_offer*1.15
        print(WRs['player'].item() + ' sends the following counter offer...')
        print(round(counter_offer, -5))
    elif (original_accept > 60) & (original_accept <= 80):
        counter_offer = user_contract_offer*1.1
        print(WRs['player'].item() + ' sends the following counter offer...')
        print(round(counter_offer, -5))
    else:
        counter_offer = user_contract_offer*1.05
        print(WRs['player'].item() + ' sends the following counter offer...')
        print(round(counter_offer, -5))
    print()
    print('Does GM accepet the counter offer?(yes/no)')
    GM_Decision = input()
    if GM_Decision == 'yes':
        print('Congrats on signing ' + WRs['player'].item() + ' to your team!')
    elif GM_Decision == 'no':
        print('Sorry you could not get a deal done. You may try starting a new negotiation or move on')
    else:
        print('Invalid input. Start negotiations over.')
else:
    print('ERROR')

#for counter