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

def concat_api(source, first_season, last_season):
    df_list = []
    for i in range(first_season, last_season+1):
        temp = pull_api(source %i)
        df_list.append(temp)
    df = pd.concat(df_list)
    return df

#### WAR Data ####
#pff_war = pull_s3("iq/nfl/2021/war.csv.gz")
pff_war = pull_s3("iq/nfl/%i/war.csv.gz", season_start=2011, season_end=2021)
war_cols = [
    'player_id',
    'player',
    'season',
    'year_in_league',
    'season_count',
    'war',
    'waa'
]
pff_war = pff_war[war_cols]
#aggregate to account for playes who played on multiple teams in 2021
pff_war = pff_war.groupby(['player_id', 'player', 'season'], as_index=False).mean()

#### PFF Grade Data ####
#pff_grades = pull_api('/v1/grades/nfl/2021/season_grade')
pff_grades = concat_api('/v1/grades/nfl/%i/season_grade', 2011, 2021)
#player_id = pff_id
grade_cols = [
    'player_id',
    'position',
    'unit',
    'season',
    #'discipline',
    'offense',
    'pass_block',
    'run_block',
    'receiving',
    'pass',
    'run',
    'defense',
    'coverage',
    'pass_rush',
    'run_defense'
]
pff_grades = pff_grades[grade_cols]
#aggregate to season grade
pff_grades = pff_grades.groupby(['player_id', 'position', 'unit', 'season'], as_index=False).mean()

#### Player Measurement Data ####
'''player_measurements = pull_s3("master_data/players/export.csv.gz")
measurement_cols = [
    'id',
    'height',
    'weight'
]
player_measurements = player_measurements[measurement_cols].rename(columns={'id': 'player_id'})'''

player_measurements = pull_api("/v1/player_combine_results")
measurement_cols = [
    'player_id',
    'height_in_inches',
    'weight_in_pounds',
    'wingspan_in_inches',
    'arm_length_in_inches',
    'right_hand_size_in_inches',
    'left_hand_size_in_inches',
    'fourty_time_in_seconds',
    'twenty_time_in_seconds',
    'ten_time_in_seconds',
    'twenty_shuttle_in_seconds',
    'three_cone_in_seconds',
    'vertical_jump_in_inches',
    'broad_jump_in_inches',
    'bench_press_in_reps'
]
player_measurements = player_measurements[measurement_cols]

#### Snap Count Data ####
#snaps = pull_api('/v1/fantasy/nfl/2021/snaps')
snaps = concat_api('/v1/fantasy/nfl/%i/snaps', 2011, 2021)
snaps = snaps[snaps['week'] > 0] #only want regular season and playoff snaps
snaps['total_alignments_played'] = snaps.groupby(['player_id', 'season'])['position'].transform('nunique') #count of all different alignments played during season
positions_conditions = [(snaps["position"] == "HB") | (snaps["position"] == "HB-L") | (snaps["position"] == "HB-R"),
                        (snaps["position"] == "LWR") | (snaps["position"] == "RWR"), 
                        (snaps["position"] == "SLiWR") |
                        (snaps["position"] == "SRiWR") | (snaps["position"] == "SLoWR") | (snaps["position"] == "SRoWR") |
                        (snaps["position"] == "SLWR") | (snaps["position"] == "SRWR"),
                        (snaps["position"] == "QB"), 
                        (snaps["position"] == "LT"),
                        (snaps["position"] == "LG"),
                        (snaps["position"] == "C"),
                        (snaps["position"] == "RG"),
                        (snaps["position"] == "RT"),
                        (snaps["position"] == "FB") | (snaps["position"] == "FB-iL") | (snaps["position"] == "FB-iR") |
                        (snaps["position"] == "FB-oL") | (snaps["position"] == "FB-oR") | (snaps["position"] == "FB-L") | (snaps["position"] == "FB-R"),
                        (snaps["position"] == "TE-L") | (snaps["position"] == "TE-R") | (snaps["position"] == "TE-iL") |
                        (snaps["position"] == "TE-iR") | (snaps["position"] == "TE-oL") | (snaps["position"] == "TE-oR"),
                        ((snaps['position'] == 'LEO') | (snaps['position'] == 'LOLB')),
                        ((snaps['position'] == 'REO') | (snaps['position'] == 'ROLB')),
                        ((snaps['position'] == 'LE') | (snaps['position'] == 'DLT')),
                        ((snaps['position'] == 'RE') | (snaps['position'] == 'DRT')),
                        ((snaps['position'] == 'NLT') | (snaps['position'] == 'NRT') | (snaps['position'] == 'NT')),
                        ((snaps['position'] == 'MLB') | (snaps['position'] == 'RLB') | (snaps['position'] == 'LLB') | (snaps['position'] == 'RILB') | (snaps['position'] == 'LILB')),
                        ((snaps['position'] == 'SCBiL') | (snaps['position'] == 'SCBoL') | (snaps['position'] == 'SCBiR') | (snaps['position'] == 'SCBoR') | (snaps['position'] == 'SCBL') | (snaps['position'] == 'SCBR')),
                        ((snaps['position'] == 'LCB') | (snaps['position'] == 'RCB')),
                        ((snaps['position'] == 'FS') | (snaps['position'] == 'FSR') | (snaps['position'] == 'FSL')),
                        ((snaps['position'] == 'SS') | (snaps['position'] == 'SSR') | (snaps['position'] == 'SSL'))]
positions_choices = ["HB", "WR", "SWR", "QB", "LT", "LG", "C", "RG", "RT", "FB", "TE", 
                    'LED', 'RED', 'LID', 'RID', 'NT', 'LB', 'SCB', 'CB', 'FS', 'SS']
snaps["position"] = np.select(positions_conditions, positions_choices, default=snaps['position'])
snaps['total_positions_played'] = snaps.groupby(['player_id', 'season'])['position'].transform('nunique') #total number of positions played throughout season
snaps['total_snap_count'] = snaps.groupby(['player_id', 'season'])['snap_count'].transform('sum')
snaps['weeks_played'] = snaps.groupby(['player_id', 'season'])['week'].transform('nunique')
snaps['snap_count_per_week_played'] = snaps['total_snap_count']/snaps['weeks_played']
snaps['snaps_by_position'] = snaps.groupby(['player_id', 'position', 'season'])['snap_count'].transform('sum')
for position in positions_choices:
    snaps[position+'_total_snaps'] = np.where(snaps['position'] == position, snaps['snaps_by_position'], 0)
snaps = snaps.drop_duplicates(subset=['player_id', 'position', 'season'])
for position in positions_choices:
    snaps[position+'_total_snaps'] = snaps.groupby(['player_id', 'season'])[position+'_total_snaps'].transform('sum')
snaps_cols = [
    'player_id',
    'season', 
    'total_alignments_played', 
    'total_positions_played', 
    'total_snap_count', 
    'weeks_played', 
    'snap_count_per_week_played',
    'HB_total_snaps',
    'WR_total_snaps',
    'SWR_total_snaps',
    'QB_total_snaps',
    'LT_total_snaps',
    'LG_total_snaps',
    'C_total_snaps',
    'RG_total_snaps',
    'RT_total_snaps',
    'FB_total_snaps',
    'TE_total_snaps',
    'LED_total_snaps',
    'RED_total_snaps',
    'LID_total_snaps',
    'RID_total_snaps',
    'NT_total_snaps',
    'LB_total_snaps',
    'SCB_total_snaps',
    'CB_total_snaps',
    'FS_total_snaps',
    'SS_total_snaps'
]
position_snaps_cols = [
    'HB_total_snaps',
    'WR_total_snaps',
    'SWR_total_snaps',
    'QB_total_snaps',
    'LT_total_snaps',
    'LG_total_snaps',
    'C_total_snaps',
    'RG_total_snaps',
    'RT_total_snaps',
    'FB_total_snaps',
    'TE_total_snaps',
    'LED_total_snaps',
    'RED_total_snaps',
    'LID_total_snaps',
    'RID_total_snaps',
    'NT_total_snaps',
    'LB_total_snaps',
    'SCB_total_snaps',
    'CB_total_snaps',
    'FS_total_snaps',
    'SS_total_snaps'
]

snaps = snaps[snaps_cols]
snaps = snaps.drop_duplicates()
snaps['main_position'] = snaps[position_snaps_cols].idxmax(axis=1)
snaps['main_position'] = snaps['main_position'].apply(lambda x: str(x).removesuffix('_total_snaps'))

#### Game Stats Data ####
#TODO add game stats like targets, routes run, targets per route, receptions per target, routes per snap, yards per target, yards per catch, YAC, Route Depth, etc.
rushing_fantasy = concat_api('/v1/fantasy/nfl/%i/rushing', 2011, 2021)
#only regular season stats
rushing_totals_season = rushing_fantasy.drop(rushing_fantasy[(rushing_fantasy['season'] >= 2021) & (rushing_fantasy['week'] > 18)].index).reset_index(drop=True)
rushing_totals_season = rushing_totals_season.drop(rushing_totals_season[(rushing_totals_season['season'] < 2021) & (rushing_totals_season['week'] > 17)].index).reset_index(drop=True)
rushing_totals_season = rushing_totals_season.drop(rushing_totals_season[(rushing_totals_season['week'] < 1)].index).reset_index(drop=True)
rushing_totals_season = rushing_totals_season.groupby(['rusher_player_id', 'season'], as_index=False)[['yards', 'attempt', 'touchdown']].sum()
rushing_totals_season = rushing_totals_season.rename(columns={'yards':'rushing_yards', 
                                                            'rusher_player_id':'player_id',
                                                            'attempt': 'rush_attempt',
                                                            'touchdown': 'rush_touchdown'})

receiving_fantasy = concat_api('/v1/fantasy/nfl/%i/receiving', 2011, 2021)
#only regular season stats
receiving_totals_season = receiving_fantasy.drop(receiving_fantasy[(receiving_fantasy['season'] >= 2021) & (receiving_fantasy['week'] > 18)].index).reset_index(drop=True)
receiving_totals_season = receiving_totals_season.drop(receiving_totals_season[(receiving_totals_season['season'] < 2021) & (receiving_totals_season['week'] > 17)].index).reset_index(drop=True)
receiving_totals_season = receiving_totals_season.drop(receiving_totals_season[(receiving_totals_season['week'] < 1)].index).reset_index(drop=True)
receiving_totals_season = receiving_totals_season.groupby(['target_player_id', 'season'], as_index=False)[['yards', 'reception', 'target', 'touchdown']].sum()
receiving_totals_season = receiving_totals_season.rename(columns={'yards':'receiving_yards', 
                                                                'target_player_id':'player_id',
                                                                'touchdown': 'receiving_touchdown'})

passing_fantasy = concat_api('/v1/fantasy/nfl/%i/passing', 2011, 2021)
#only regular season stats
passing_totals_season = passing_fantasy.drop(passing_fantasy[(passing_fantasy['season'] >= 2021) & (passing_fantasy['week'] > 18)].index).reset_index(drop=True)
passing_totals_season = passing_totals_season.drop(passing_totals_season[(passing_totals_season['season'] < 2021) & (passing_totals_season['week'] > 17)].index).reset_index(drop=True)
passing_totals_season = passing_totals_season.drop(passing_totals_season[(passing_totals_season['week'] < 1)].index).reset_index(drop=True)
passing_totals_season = passing_totals_season.groupby(['passer_player_id', 'season'], as_index=False)[['yards', 'completion', 'attempt', 'touchdown', 'interception']].sum()
passing_totals_season = passing_totals_season.rename(columns={'yards':'pass_yards', 
                                                                'passer_player_id':'player_id',
                                                                'completion': 'pass_completion',
                                                                'attempt': 'pass_attempt',
                                                                'touchdown': 'pass_touchdown'})

pass_blocking = concat_api('/v1/analytics/projections/by_facet/nfl/%i/pass_blocking', 2011, 2021)
#only regular season stats
pass_blocking_totals_season = pass_blocking.drop(pass_blocking[(pass_blocking['season'] >= 2021) & (pass_blocking['week'] > 18)].index).reset_index(drop=True)
pass_blocking_totals_season = pass_blocking_totals_season.drop(pass_blocking_totals_season[(pass_blocking_totals_season['season'] < 2021) & (pass_blocking_totals_season['week'] > 17)].index).reset_index(drop=True)
pass_blocking_totals_season = pass_blocking_totals_season.drop(pass_blocking_totals_season[(pass_blocking_totals_season['week'] < 1)].index).reset_index(drop=True)
pass_blocking_totals_season = pass_blocking_totals_season.groupby(['player_id', 'season'], as_index=False)[['pressure_allowed', 'hit_allowed', 'hurry_allowed', 'sack_allowed']].sum()

defense_stats = concat_api('/v1/fantasy/nfl/%i/defense', 2011, 2021)
#only regular season stats
defense_totals_season = defense_stats.drop(defense_stats[(defense_stats['season'] >= 2021) & (defense_stats['week'] > 18)].index).reset_index(drop=True)
defense_totals_season = defense_totals_season.drop(defense_totals_season[(defense_totals_season['season'] < 2021) & (defense_totals_season['week'] > 17)].index).reset_index(drop=True)
defense_totals_season = defense_totals_season.drop(defense_totals_season[(defense_totals_season['week'] < 1)].index).reset_index(drop=True)
defense_totals_season = defense_totals_season.groupby(['player_id', 'season'], as_index=False)[['hurries', 'pressures', 'sacks', 'forced_fumble', 'stops', 'tackle', 'assisted_tackle', 'interceptions', 'missed_tackle', 'pass_breakups', 'penalty', 'fumble_recovery', 'hits', 'batted_pass']].sum()

pff_ids = pd.read_excel('../../data/HistoricalEarnings.xlsx')
draft_round = pff_ids.copy()
draft_round = draft_round.drop(columns=['player_id'])
draft_round = draft_round.rename(columns={'pff_id':'player_id'})
draft_round = draft_round[['Name', 'player_id', 'Draft Round']]
draft_round = draft_round.drop_duplicates()
draft_round = draft_round.dropna(subset=['Draft Round'])
draft_round = draft_round[draft_round['Draft Round'] != 0]

#### Merge Data ####
# merge WAR, PFF Grades, player measurements, and snaps
players = pd.merge(pff_grades, pff_war, on=['player_id', 'season'], how='left')
players = pd.merge(players, player_measurements, on='player_id', how='left')
players = pd.merge(players, snaps, on=['player_id', 'season'], how='left')
players = pd.merge(players, rushing_totals_season, on=['player_id', 'season'], how='left')
players = pd.merge(players, receiving_totals_season, on=['player_id', 'season'], how='left')
players = pd.merge(players, passing_totals_season, on=['player_id', 'season'], how='left')
players = pd.merge(players, pass_blocking_totals_season, on=['player_id', 'season'], how='left')
players = pd.merge(players, defense_totals_season, on=['player_id', 'season'], how='left')
players = pd.merge(players, draft_round, on='player_id', how='left')

players.to_csv('player_data.csv', index=False)

'''
#reduce to WRs for WR model
WRs = players[players['position'] == 'WR']
Safeties = players[players['position'] == 'S']
IOLs = players[(players['position'] == 'G') | (players['position'] == 'C')]


# save wr data needed for clustering and building contracts.
WRs.to_csv('wr_data.csv', index=False)
Safeties.to_csv('safety_data.csv', index=False)
IOLs.to_csv('iol_data.csv', index=False)
'''

#contracts = pd.read_csv('../../data/FullContracts2013-21.csv')
contracts = pd.read_excel('../../data/FullContracts2013-22.xlsx')
#cap = pd.read_excel('../../data/Cap2013-2021.xlsx')
cap = pd.read_excel('../../data/Cap2013-2022.xlsx')

contracts = contracts.rename(columns={'APY as % of Cap': 'APY%'})

contract_cols = ['player_id', 'Name', 'ID', 'Total', 'APY', 'APY%', 'Guarantee', 'contract_type', 'year_signed', 'Position']
cap_cols = ['player_id', 'contract_id', 'year', 'base_salary']

contracts = contracts[contract_cols].rename(columns={'ID': 'contract_id'})
cap = cap.rename(columns={'BS': 'base_salary', 'Year': 'year'})
cap = cap[cap_cols]

contracts = pd.merge(contracts, cap, left_on=['player_id', 'contract_id', 'year_signed'], right_on=['player_id', 'contract_id', 'year'], how='left').dropna()

contracts['contract_year_1'] = contracts['year_signed'] - 1
contracts['contract_year_2'] = contracts['year_signed'] - 2

#WRContracts = contracts[contracts['Position'] == 'WR']

#pff_ids = pd.read_excel('../../data/HistoricalEarnings.xlsx')
#pff_ids = pff_ids.rename(columns={'player_id':'pff_id', 'non_pff_id':'player_id'})
pff_ids = pff_ids[['Name', 'pff_id', 'player_id']]
pff_ids = pff_ids.drop_duplicates()

contracts = pd.merge(contracts, pff_ids[['player_id', 'pff_id']], on='player_id', how='left')

sfa_minimum_dict = {
    2013: 715000,
    2014: 730000,
    2015: 745000,
    2016: 760000,
    2017: 775000,
    2018: 790000,
    2019: 805000,
    2020: 910000,
    2021: 990000,
    2022: 1035000
}

contracts = contracts[contracts['contract_type'] != 'Drafted'] #remove rookie contracts
contracts = contracts[contracts['contract_type'] != 'UDFA'] #remove undrafted free agents contracts
contracts = contracts[contracts['contract_type'] != 'Practice'] #remove practice squad players contracts
contracts = contracts[contracts['contract_type'] != 'RFA'] #remove restriced free agent contracts
#contracts = contracts[(contracts['base_salary'] > 990000) | (contracts['contract_type'] != 'SFA')]
for year, minimum in sfa_minimum_dict.items():
    contracts = contracts.drop(contracts[(contracts['year_signed'] == year) & (contracts['base_salary'] < minimum) & (contracts['contract_type'] == 'SFA')].index)

contracts['Position'] = contracts['Position'].str.upper()

contract_pos_conditions = [(contracts["Position"] == "LT") | (contracts["Position"] == "RT"),
                            (contracts["Position"] == "LG") | (contracts["Position"] == "RG"),
                            (contracts["Position"] == "RB"),
                            (contracts["Position"] == "EDGE") | (contracts["Position"] == "34OLB") | (contracts["Position"] == "DE"),
                            (contracts["Position"] == "DT") | (contracts["Position"] == "43DT") | (contracts["Position"] == "DI"),
                            (contracts["Position"] == "ILB")]
contract_pos_choices = ['T', 'G', 'HB', 'ED', 'DL', 'LB']
contracts["Position"] = np.select(contract_pos_conditions, contract_pos_choices, default=contracts['Position'])

salary_cap_dict = {
	2013: 123600000,
	2014: 133000000,
	2015: 143280000,
	2016: 155270000,
	2017: 167000000,
	2018: 177200000,
	2019: 188200000,
	2020: 198200000,
	2021: 182500000, # maybe change this to 200 mil? since it's an anomaly?
	2022: 208200000
}
current_cap = 208200000

contracts['APY_Adj'] = 0
for year, cap in salary_cap_dict.items():
    contracts['APY_Adj'] = np.where(contracts['year_signed'] == year, contracts['APY']*(current_cap/cap), contracts['APY_Adj'])
    

contracts.to_csv('contract_data.csv', index=False)

'''
#### WR Contracts ####

WRContracts = contracts[contracts['Position'] == 'WR']

#WRContracts = pd.merge(WRContracts, pff_ids[['player_id', 'pff_id']], on='player_id', how='left')

WRContracts = WRContracts.drop(columns=['player_id'])
WRContracts = WRContracts.rename(columns={'pff_id':'player_id'})
WRContracts = WRContracts[WRContracts['player_id'].notna()]
WRContracts['player_id'] = WRContracts['player_id'].astype('int64')

WRs_y1 = WRs
WRs_y2 = WRs.copy()

WRs_y1 = WRs_y1.add_suffix('_y1')
WRs_y2 = WRs_y2.add_suffix('_y2')

WRs_y1 = WRs_y1.rename(columns={'player_id_y1': 'player_id', 'season_y1': 'contract_year_1'})
WRs_y2 = WRs_y2.rename(columns={'player_id_y2': 'player_id', 'season_y2': 'contract_year_2'})

WRContracts = pd.merge(WRContracts, WRs_y1, on=['player_id', 'contract_year_1'], how='left')
WRContracts = pd.merge(WRContracts, WRs_y2, on=['player_id', 'contract_year_2'], how='left')

WRContracts.to_csv('wr_contract_data.csv', index=False)

#### IOL Contracts ####

IOLContracts = contracts[(contracts['Position'] == 'G') | (contracts['Position'] == 'C')]

IOLContracts = IOLContracts.drop(columns=['player_id'])
IOLContracts = IOLContracts.rename(columns={'pff_id':'player_id'})
IOLContracts = IOLContracts[IOLContracts['player_id'].notna()]
IOLContracts['player_id'] = IOLContracts['player_id'].astype('int64')

IOL_y1 = IOLs
IOL_y2 = IOLs.copy()

IOL_y1 = IOL_y1.add_suffix('_y1')
IOL_y2 = IOL_y2.add_suffix('_y2')

IOL_y1 = IOL_y1.rename(columns={'player_id_y1': 'player_id', 'season_y1': 'contract_year_1'})
IOL_y2 = IOL_y2.rename(columns={'player_id_y2': 'player_id', 'season_y2': 'contract_year_2'})

IOLContracts = pd.merge(IOLContracts, IOL_y1, on=['player_id', 'contract_year_1'], how='left')
IOLContracts = pd.merge(IOLContracts, IOL_y2, on=['player_id', 'contract_year_2'], how='left')

IOLContracts.to_csv('iol_contract_data.csv', index=False)

#### Safety Contracts ####

SContracts = contracts[contracts['Position'] == 'S']

SContracts = SContracts.drop(columns=['player_id'])
SContracts = SContracts.rename(columns={'pff_id':'player_id'})
SContracts = SContracts[SContracts['player_id'].notna()]
SContracts['player_id'] = SContracts['player_id'].astype('int64')

S_y1 = Safeties
S_y2 = Safeties.copy()

S_y1 = S_y1.add_suffix('_y1')
S_y2 = S_y2.add_suffix('_y2')

S_y1 = S_y1.rename(columns={'player_id_y1': 'player_id', 'season_y1': 'contract_year_1'})
S_y2 = S_y2.rename(columns={'player_id_y2': 'player_id', 'season_y2': 'contract_year_2'})

SContracts = pd.merge(SContracts, S_y1, on=['player_id', 'contract_year_1'], how='left')
SContracts = pd.merge(SContracts, S_y2, on=['player_id', 'contract_year_2'], how='left')

SContracts.to_csv('safety_contract_data.csv', index=False)
'''

## think of of mathematical review of this contract stuff