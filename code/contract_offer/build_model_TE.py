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
#player_id = 11784 #David Njoku
player_id = 9637 #Darren Waller
#player_id = 47304 #Dalton Schultz
#player_id = 47153 #Mike Gesicki


#TEs = pd.read_csv('TE_data.csv')
players = pd.read_csv('player_data.csv')
TEs = players[players['position'] == 'TE']
contracts = pd.read_csv('contract_data.csv')

'''receiving_fantasy = concat_api('/v1/fantasy/nfl/%i/receiving', 2011, 2021)
#TODO seasons before 2021 go up to week 17, others week 18
receiving_totals_season = receiving_fantasy[receiving_fantasy['week'] < 19]
receiving_totals_season = receiving_totals_season.groupby(['target_player_id', 'season'], as_index=False)[['yards', 'reception', 'target', 'touchdown']].sum()
receiving_totals_season = receiving_totals_season.rename(columns={'yards':'receiving_yards', 'target_player_id':'player_id'})

TEs = pd.merge(TEs, receiving_totals_season, on=['player_id', 'season'], how='left')'''

TEContracts = contracts[contracts['Position'] == 'TE']

#TEContracts = pd.merge(TEContracts, pff_ids[['player_id', 'pff_id']], on='player_id', how='left')

TEContracts = TEContracts.drop(columns=['player_id'])
TEContracts = TEContracts.rename(columns={'pff_id':'player_id'})
TEContracts = TEContracts[TEContracts['player_id'].notna()]
TEContracts['player_id'] = TEContracts['player_id'].astype('int64')

TEs_y1 = TEs.copy()
TEs_y2 = TEs.copy()

TEs_y1 = TEs_y1.add_suffix('_y1')
TEs_y2 = TEs_y2.add_suffix('_y2')

TEs_y1 = TEs_y1.rename(columns={'player_id_y1': 'player_id', 'season_y1': 'contract_year_1'})
TEs_y2 = TEs_y2.rename(columns={'player_id_y2': 'player_id', 'season_y2': 'contract_year_2'})

TEContracts = pd.merge(TEContracts, TEs_y1, on=['player_id', 'contract_year_1'], how='left')
TEContracts = pd.merge(TEContracts, TEs_y2, on=['player_id', 'contract_year_2'], how='left')

#TEContracts = pd.read_csv('TE_contract_data.csv')
TEContracts['nameAndContractYear'] = TEContracts['Name'] + '_' + TEContracts['year_signed'].astype('str')
TEContracts = TEContracts.drop_duplicates(subset=['nameAndContractYear'])
TEContracts['grade'] = TEContracts[['offense_y1', 'offense_y2']].mean(axis=1)
TEContracts['WAR'] = TEContracts[['war_y1', 'war_y2']].mean(axis=1)
TEContracts['receiving_grade'] = TEContracts[['receiving_y1', 'receiving_y2']].mean(axis=1)
TEContracts['receiving_yards'] = TEContracts[['receiving_yards_y1', 'receiving_yards_y2']].sum(axis=1)
TEContracts['receptions'] = TEContracts[['reception_y1', 'reception_y2']].sum(axis=1)
TEContracts['targets'] = TEContracts[['target_y1', 'target_y2']].sum(axis=1)
TEContracts['receiving_touchdowns'] = TEContracts[['receiving_touchdown_y1', 'receiving_touchdown_y2']].sum(axis=1)
TEContracts['WAR_Percentile'] = TEContracts['WAR'].rank(pct=True)
TEContracts['draft_round'] = TEContracts['Draft Round_y1'].fillna(0).astype('int')

TEs = TEs[TEs['player_id'] == player_id]

#TEs['year_signed'] = contract_year
#TEs['contract_year_1'] = TEs['year_signed'] - 1
#TEs['contract_year_2'] = TEs['year_signed'] - 2

TEs_y1 = TEs[TEs['season'] == contract_year - 1]
TEs_y2 = TEs[TEs['season'] == contract_year - 2]

TEs_y1 = TEs_y1.add_suffix('_y1')
TEs_y2 = TEs_y2.add_suffix('_y2')

TEs_y1 = TEs_y1.rename(columns={'player_id_y1': 'player_id', 'season_y1': 'contract_year_1'})
TEs_y2 = TEs_y2.rename(columns={'player_id_y2': 'player_id', 'season_y2': 'contract_year_2'})

#TEs['year_signed'] = contract_year
TEs['contract_year_1'] = contract_year - 1
TEs['contract_year_2'] = contract_year - 2
TEs = TEs[['player_id', 'contract_year_1', 'contract_year_2']]
TEs = TEs.drop_duplicates()

player_info = pd.merge(TEs, TEs_y1, on=['player_id', 'contract_year_1'], how='left')
player_info = pd.merge(player_info, TEs_y2, on=['player_id', 'contract_year_2'], how='left')
player_info['grade'] = player_info[['offense_y1', 'offense_y2']].mean(axis=1)
player_info['WAR'] = player_info[['war_y1', 'war_y2']].mean(axis=1)
player_info['receiving_grade'] = player_info[['receiving_y1', 'receiving_y2']].mean(axis=1)
player_info['receiving_yards'] = player_info[['receiving_yards_y1', 'receiving_yards_y2']].sum(axis=1)
player_info['receptions'] = player_info[['reception_y1', 'reception_y2']].sum(axis=1)
player_info['targets'] = player_info[['target_y1', 'target_y2']].sum(axis=1)
player_info['receiving_touchdowns'] = player_info[['receiving_touchdown_y1', 'receiving_touchdown_y2']].sum(axis=1)
player_info['WAR_Percentile'] = stats.percentileofscore(TEContracts['WAR'], player_info['WAR'][0])/100
player_info['draft_round'] = player_info['Draft Round_y1'].fillna(0).astype('int')

player_grade = player_info['offense_y1'].values[player_info['player_id'] == player_id][0]

offense_cluster_cols = [
    'grade',
    'WAR',
    'receiving_grade',
    'receiving_yards',
    'receptions',
    'targets',
    'receiving_touchdowns',
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
    'TE_total_snaps_y1',
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
    'TE_total_snaps_y2'
]

X = TEContracts[offense_cluster_cols].fillna(0)
kmeanModel = KMeans(n_clusters=5)#.fit(X)
kmeanModel.fit(X)
TEContracts['cluster'] = kmeanModel.labels_

player_cluster = kmeanModel.predict(player_info[offense_cluster_cols])


TEContracts['playerComp'] = TEContracts['offense_y1'].sub(player_grade).abs()
top_10_similar = TEContracts[TEContracts['cluster'] == player_cluster[0]].nsmallest(10, 'playerComp') #should be top 10 similar players plus player to compare
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
    #'year_in_league_y1',
    #'total_alignments_played_y1',
    #'total_positions_played_y1',
    #'total_snap_count_y1',
    #'weeks_played_y1',
    #'snap_count_per_week_played_y1',
    #'TE_total_snaps_y1',
    #'STE_total_snaps_y1',
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
    #'TE_total_snaps_y2',
    #'STE_total_snaps_y2'
    'draft_round'
]
#y_col = 'APY%'
y_col = 'APY_Adj'

'''similar_players = TEContracts[TEContracts['cluster'] == player_cluster[0]]
similar_players = similar_players.dropna(subset=offense_regression_cols)
X = similar_players[offense_regression_cols]
y = similar_players[y_col]'''

TEContracts = TEContracts.dropna(subset=offense_regression_cols)
X = TEContracts[offense_regression_cols]
y = TEContracts[y_col]

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

#finish all contract offer models for every position
#refine all contract offer models with more positional data (i.e. receiving yards, rushing yards, etc.)
#have a strong decision model for at least one position finished

# look into why years spike in contract
# i.e. a lot of high war players entering FA in the next year
# premium vs non premium position
# take a look at max median and mean WAR

# for rookies and 1-2 rounders look at two best years
# for players leaving rookie deals who were 1-2 rounders, look at 2 best years