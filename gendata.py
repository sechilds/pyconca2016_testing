from itertools import zip_longest
import pandas as pd
import numpy as np

def get_stations(state):
    subway_stations = ['Bathurst', 'Bay', 'Bayview', 'Bessarion', 'Bloor-Yonge',
                       'Broadview', 'Castle Frank', 'Chester', 'Christie', 'College',
                       'Coxwell', 'Davisville', 'Don Mills', 'Donlands',
                       'Downsview', 'Dufferin', 'Dundas', 'Dundas West',
                       'Dupont', 'Eglinton', 'Eglinton West', 'Ellesmere', 'Finch',
                       'Glencairn', 'Greenwood', 'High Park', 'Islington', 'Jane',
                       'Keele', 'Kennedy', 'King', 'Kipling', 'Lansdowne',
                       'Lawrence', 'Lawrence East', 'Lawrence West', 'Leslie',
                       'Main Street', 'McCowan', 'Midland', 'Museum',
                       'North York Centre', 'Old Mill', 'Osgoode', 'Ossington',
                       'Pape', 'Queen', "Queen's Park", 'Rosedale', 'Royal York',
                       'Runnymede', 'Scarborough Centre', 'Sheppard-Yonge',
                       'Sherbourne', 'Spadina', 'St Andrew', 'St Clair',
                       'St Clair West', 'St George', 'St Patrick', 'Summerhill',
                       'Union', 'Victoria Park', 'Warden', 'Wellesley', 'Wilson',
                       'Woodbine', 'York Mills', 'Yorkdale'] 
    state.shuffle(subway_stations)
    return subway_stations

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def get_cols(state):
    return grouper(get_stations(state), 20)

def original_data(cols, state, makenan=False):
    x =  pd.DataFrame(state.randint(1,25,size=(100,20)),
                        index=range(100), columns=cols.__next__())
    if makenan:
        x = x.applymap(lambda x: np.nan if x > 22 else x)
    return x

def modified_data(x, cols, state, makenan=False):
    y = x.drop(x.columns[np.concatenate((state.randint(0,4,size=1),
                                         state.randint(5,19,size=4)))],
               axis=1)
    new_cols = pd.DataFrame(state.randint(1,25,size=(100,20)),
                            index=range(100), columns=cols.__next__())
    y = pd.concat([y, new_cols], axis=1)
    new_rows = pd.DataFrame(state.randint(1,25,size=(20,y.shape[1])),
                            index=range(101,121), columns=y.columns)
    y = pd.concat([y, new_rows], axis=0)
    y = y.drop(y.index[np.concatenate((state.randint(0,4,size=1),
                                       state.randint(5,120,size=4)))],axis=0)
    change_rows = np.concatenate((state.randint(0,4,size=1),
                                  state.randint(5,y.shape[0],size=19)))
    change_cols = np.concatenate((state.randint(0,4,size=1),
                                  state.randint(5,y.shape[1],size=19)))
    new_coords = np.stack([change_rows, change_cols], axis=1)
    for i in new_coords:
        if ~np.isnan(y.iloc[i[0], i[1]]):
            y.iloc[i[0], i[1]] = y.iloc[i[0], i[1]] + state.randint(-10,10,size=1)
    if makenan:
        y = y.applymap(lambda x: np.nan if x > 22 else x)
    return y

