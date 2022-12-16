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


positions = ['WR', 'QB', 'HB']
players = pd.read_csv('player_data.csv')
players = players[['player_id', 'player', 'position']].drop_duplicates()

def make_window():
    sg.theme('DarkBlack')

    layout = [
        [sg.Text('Contract Negotiation')],
        [sg.Text('Position:'), sg.OptionMenu(values=(positions), k='-POSITION-')],
        [sg.Button('Choose Position')],
        [sg.Text('Player:'), sg.OptionMenu(values=(list(players['player'].unique())), k='-PLAYER-'),],
        [sg.Button('Choose Player')],
        [sg.Text('Offer Suggestions:')],
        [sg.Multiline(size=(60,5), font='Courier 8', write_only=True, k='-SUGGESTIONS-')],
        [sg.Text('Contract Offer:'), sg.Input(key='-OFFER-')],
        [sg.Button('Send Offer')],
        [sg.Text('Player Response:')],
        [sg.Multiline(size=(60,5), font='Courier 8', expand_x=True, expand_y=True, write_only=True,
                                    reroute_stdout=True, reroute_stderr=True, echo_stdout_stderr=True, autoscroll=True, auto_refresh=True)],
        [sg.Button('Accept'), sg.Button('Counter'), sg.Button('Decline')]
    ]
    window = sg.Window('PFF Mock FA', layout)
    
    return window

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
            free_agent = players[players['player'] == values['-PLAYER-']]
            #print(event, values['-POSITION-'], values['-PLAYER-'])
            #print(free_agent)
            window['-SUGGESTIONS-'].update(value=free_agent.iloc[0]['player_id'])
            window['-SUGGESTIONS-'].update(value='\nblah', append=True)
    
    window.close()

if __name__ == '__main__':
    main()