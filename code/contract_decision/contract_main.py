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
from build_model_WR import *
import PySimpleGUI as sg
#import PySimpleGUIWeb as sg

#ensure correct file path
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
print(dname)
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

#### constants ####
positions = ['WR', 'QB', 'RB']
contract_year = 2023
#players = pd.read_csv('player_data.csv')
#players = pd.read_csv('../../data/FA_rankings_2023.csv')
players = pd.read_csv('FA_rankings_2023.csv')
players = players.rename(columns={'PFFID': 'player_id', 'Name': 'player', 'Position': 'position'})
players = players[['player_id', 'player', 'position']].drop_duplicates()
players = players.sort_values(by=['player'])

#contract_year = 2022
#print('enter player id')
#player_id = int(input())
#print('enter position')
#position = input()

#players = players[players['player_id'] == player_id] 
def make_window():
    sg.theme('DarkBlack')

    layout = [
        [sg.Text('Contract Negotiation')],
        [sg.Text('Position:'), sg.Combo(values=(positions), key='-POSITION-')],
        [sg.Button('Choose Position')],
        [sg.Text('Player:'), sg.Combo(values=(list(players['player'].unique())), key='-PLAYER-'),],
        [sg.Button('Choose Player')],
        [sg.Text('Offer Suggestions:')],
        [sg.Multiline(size=(60,5), font='Courier 8',  key='-SUGGESTIONS-')], #write_only=True,
        [sg.Text('Contract Offer:'), sg.Input(key='-OFFER-')],
        [sg.Button('Send Offer')],
        [sg.Text('Player Response:')],
        [sg.Multiline(size=(60,5), font='Courier 8',  key='-DECISION-')], #write_only=True,
        [sg.Button('Accept'), sg.Button('Counter'), sg.Button('Decline')]
    ]
    window = sg.Window('PFF Mock FA', layout)
    
    return window

def contract_negotiation(free_agent, lower_bound, upper_bound, offer):
    #### call contract scripts to get bounds and contract options depending on player and position ####
    #if position == 'WR':
    #    lower_bound, option1, option2, option3, upper_bound = WR_Contracts(contract_year, player_id)

    #### use contract scripts returned values to input into negotiation
    #print('Enter APY contract offer')
    #user_contract_offer = int(input())

    #build percentiles based on low and high bound from model
    contract_percentile_list = np.linspace(lower_bound, upper_bound, 1000)

    accept = stats.percentileofscore(contract_percentile_list, offer)
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
    #print()

    import random
    choiceList = ['accept', 'decline', 'counter']

    #print('Player Decision...')
    decision = random.choices(choiceList, weights=(accept, decline, counter))
    #decision = ['counter']
    #print(decision)

    #print()

    if decision == ['accept']:
        return ('Congrats on signing ' + free_agent['player'].item() + ' to your team!')
    elif decision == ['decline']:
        return ('Sorry, no deal.')
    elif decision == ['counter']:
        if (original_accept > 20) & (original_accept <= 40):
            counter_offer = offer*1.2
            return (free_agent['player'].item() + ' sends the following counter offer: $' + str(counter_offer))#, (round(counter_offer, -5))
        elif (original_accept > 40) & (original_accept <= 60):
            counter_offer = offer*1.15
            return (free_agent['player'].item() + ' sends the following counter offer: $' + str(counter_offer))
            #print(round(counter_offer, -5))
        elif (original_accept > 60) & (original_accept <= 80):
            counter_offer = offer*1.1
            return (free_agent['player'].item() + ' sends the following counter offer: $' + str(counter_offer))
            #print(round(counter_offer, -5))
        else:
            counter_offer = offer*1.05
            return (free_agent['player'].item() + ' sends the following counter offer: $' + str(counter_offer))
            #print(round(counter_offer, -5))
        #print()
        #print('Does GM accepet the counter offer?(yes/no)')
        #GM_Decision = input()
        #if GM_Decision == 'yes':
        #    print('Congrats on signing ' + players['player'].item() + ' to your team!')
        #elif GM_Decision == 'no':
        #    print('Sorry you could not get a deal done. You may try starting a new negotiation or move on')
        #else:
        #    print('Invalid input. Start negotiations over.')
    else:
        return ('ERROR')

def main():
    window = make_window()

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED: # if user closes window or clicks cancel
            break 
        elif event == 'Choose Position':
            players_filtered = players[players['position'] == values['-POSITION-']]
            window.Element('-PLAYER-').Update(values=list(players_filtered['player']))
        elif event == 'Choose Player':
            free_agent = players[(players['player'] == values['-PLAYER-']) & (players['position'] == values['-POSITION-'])]
            player_id = free_agent.iloc[0]['player_id']
            position = values['-POSITION-']
            if position == 'WR':
                lower_bound, option1, option2, option3, upper_bound = WR_Contracts(contract_year, player_id)
            #print(event, values['-POSITION-'], values['-PLAYER-'])
            #print(free_agent)
            window['-SUGGESTIONS-'].update(value=str(option1))
            window['-SUGGESTIONS-'].update(value='\n'+str(option2), append=True)
            window['-SUGGESTIONS-'].update(value='\n'+str(option3), append=True)
        elif event == 'Send Offer':
            offer = int(values['-OFFER-'])
            #free_agent = players[(players['player'] == values['-PLAYER-']) & (players['position'] == values['-POSITION-'])]
            answer = contract_negotiation(free_agent, lower_bound, upper_bound, offer)
            window['-DECISION-'].update(value=answer)
    
    window.close()

if __name__ == '__main__':
    main()
