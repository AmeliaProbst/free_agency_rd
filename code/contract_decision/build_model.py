import pandas as pd
import numpy as np
import os
import requests
import re
import s3fs
from dotenv import load_dotenv

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

contract_year = 2022
#player_id = 47546 #Christian Kirk
player_id = 11824 #Cooper Kupp
#player_id = 48236 #Darnell Mooney
#player_id = 33441 #Diontae Johnson
#player_id = 48274 #Deebo Samuel
#player_id = 48327 #AJ Brown

#WRs = pd.read_csv('wr_data.csv')
#WRs = pd.read_csv('wr_contract_data.csv')
#cap21 = pd.read_csv('../../../data/2021Cap.csv')
#contracts21 = pd.read_excel('../../../data/2021ContractData.xlsx')

#WRs = pd.merge(WRs, contracts21[['pff_id', 'ID', 'Total', 'APY', 'Guarantee', 'contract_type']].rename(columns={'pff_id': 'player_id', 'ID':'contract_id'}), on='player_id', how='left')
#WRs = pd.merge(WRs, cap21[['player_id', 'contract_id', 'year', 'base_salary']].rename(columns={'year':'season'}), on=['player_id', 'contract_id', 'season'], how='left')

#free_agent = WRs[WRs['player_id'] == player_id]


#cap21 = pd.read_csv('../../../data/2021Cap.csv')
#contracts21 = pd.read_excel('../../../data/2021ContractData.xlsx')
WRContracts = pd.read_csv('wr_contract_data.csv')
WRContracts['nameAndContractYear'] = WRContracts['Name'] + '_' + WRContracts['year_signed'].astype('str')
WRContracts = WRContracts.drop_duplicates(subset=['nameAndContractYear'])

WRs = pd.read_csv('wr_data.csv')
free_agent = WRs[WRs['player_id'] == player_id]

#WRs['year_signed'] = contract_year
#WRs['contract_year_1'] = WRs['year_signed'] - 1
#WRs['contract_year_2'] = WRs['year_signed'] - 2

WRs_y1 = free_agent[free_agent['season'] == contract_year - 1]
WRs_y2 = free_agent[free_agent['season'] == contract_year - 2]

WRs_y1 = WRs_y1.add_suffix('_y1')
WRs_y2 = WRs_y2.add_suffix('_y2')

WRs_y1 = WRs_y1.rename(columns={'player_id_y1': 'player_id', 'season_y1': 'contract_year_1'})
WRs_y2 = WRs_y2.rename(columns={'player_id_y2': 'player_id', 'season_y2': 'contract_year_2'})

#WRs['year_signed'] = contract_year
free_agent['contract_year_1'] = contract_year - 1
free_agent['contract_year_2'] = contract_year - 2
free_agent = free_agent[['player_id', 'contract_year_1', 'contract_year_2']]
free_agent = free_agent.drop_duplicates()

free_agent = pd.merge(free_agent, WRs_y1, on=['player_id', 'contract_year_1'], how='right')
free_agent = pd.merge(free_agent, WRs_y2, on=['player_id', 'contract_year_2'], how='right')

#filter out if APY is nan
WRContracts = WRContracts[WRContracts['APY%'].notna()]

'''
X_cols = [
    'offense',
    'year_in_league',
    'war',
    'waa',
    'total_alignments_played',
    'total_positions_played',
    'total_snap_count',
    'weeks_played',
    'snap_count_per_week_played',
    'WR_total_snaps',
    'SWR_total_snaps'
]'''
X_cols = [
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
    'WR_total_snaps_y1',
    'SWR_total_snaps_y1',
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
    'WR_total_snaps_y2',
    'SWR_total_snaps_y2'
]
y_col = 'APY%'

WRContracts = WRContracts.dropna(subset=X_cols)
X = WRContracts[X_cols]
y = WRContracts[y_col]

free_agent = free_agent[X_cols]

#from sklearn.linear_model import LinearRegression
#lr = LinearRegression()
#model = lr.fit(X, y)

from sklearn.ensemble import GradientBoostingRegressor
#build models for low middle and high
gbr_low = GradientBoostingRegressor(loss='quantile', alpha=0.1)
gbr = GradientBoostingRegressor(loss='ls')
gbr_high = GradientBoostingRegressor(loss='quantile', alpha=0.9)

model_low = gbr_low.fit(X, y)
model = gbr.fit(X, y)
model_high = gbr_high.fit(X, y)

print(model_low.predict(free_agent)[0]*208200000)
print(model.predict(free_agent)[0]*208200000)
print(model_high.predict(free_agent)[0]*208200000)

#TODO increase sample size. take players from last 5-10 years. 
# take two seasons leading up to new contract for player stats and new contract APY (single player could have multiple entries)