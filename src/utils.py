import pandas as pd
import numpy as np

usable_cols = ['AUTO STRIPPIG',
               'BACKROUND_CIRCUMSTANCES_SUSPECT_KNOWN_TO_CARRY_WEAPON_FLAG',
               'BURGLARY',
               'CPSP',
               'CRIMINAL MISCHIEF',
               'CRIMINAL POSSESSION OF CONTROLLED SUBSTANCE',
               'CRIMINAL POSSESSION OF FORGED INSTRUMENT',
               'CRIMINAL POSSESSION OF MARIHUANA',
               'CRIMINAL SALE OF CONTROLLED SUBSTANCE',
               'CRIMINAL SALE OF MARIHUANA',
               'CRIMINAL TRESPASS',
               'FIREARM_FLAG',
               'FORCIBLE TOUCHING',
               'FRISKED_FLAG',
               'GRAND LARCENY',
               'GRAND LARCENY AUTO',
               'ISSUING_OFFICER_COMMAND_CODE',
               'KNIFE_CUTTER_FLAG',
               'MAKING GRAFFITI',
               'MENACING',
               'MURDER',
               'OBSERVED_DURATION_MINUTES',
               'OFFICER_EXPLAINED_STOP_FLAG',
               'OFFICER_IN_UNIFORM_FLAG',
               'OTHER',
               'OTHER_CONTRABAND_FLAG',
               'OTHER_PERSON_STOPPED_FLAG',
               'OTHER_WEAPON_FLAG',
               'PETIT LARCENY',
               'PHYSICAL_FORCE_CEW_FLAG',
               'PHYSICAL_FORCE_DRAW_POINT_FIREARM_FLAG',
               'PHYSICAL_FORCE_HANDCUFF_SUSPECT_FLAG',
               'PHYSICAL_FORCE_OC_SPRAY_USED_FLAG',
               'PHYSICAL_FORCE_OTHER_FLAG',
               'PHYSICAL_FORCE_RESTRAINT_USED_FLAG',
               'PHYSICAL_FORCE_VERBAL_INSTRUCTION_FLAG',
               'PHYSICAL_FORCE_WEAPON_IMPACT_FLAG',
               'PROSTITUTION',
               'RAPE',
               'RECKLESS ENDANGERMENT',
               'ROBBERY',
               'SEARCH_BASIS_ADMISSION_FLAG',
               'SEARCH_BASIS_CONSENT_FLAG',
               'SEARCH_BASIS_HARD_OBJECT_FLAG',
               'SEARCH_BASIS_OTHER_FLAG',
               'SEARCH_BASIS_OUTLINE_FLAG',
               'STOP_DURATION_MINUTES',
               'STOP_WAS_INITIATED',
               'SUSPECTS_ACTIONS_CASING_FLAG',
               'SUSPECTS_ACTIONS_CONCEALED_POSSESSION_WEAPON_FLAG',
               'SUSPECTS_ACTIONS_DECRIPTION_FLAG',
               'SUSPECTS_ACTIONS_DRUG_TRANSACTIONS_FLAG',
               'SUSPECTS_ACTIONS_IDENTIFY_CRIME_PATTERN_FLAG',
               'SUSPECTS_ACTIONS_LOOKOUT_FLAG',
               'SUSPECTS_ACTIONS_OTHER_FLAG',
               'SUSPECTS_ACTIONS_PROXIMITY_TO_SCENE_FLAG',
               'TERRORISM',
               'SEARCHED_FLAG',
               'THEFT OF SERVICES',
               'UNAUTHORIZED USE OF A VEHICLE',
               'WEAPON_FOUND_FLAG',
               'ID_CARD_IDENTIFIES_OFFICER_FLAG',
               'SHIELD_IDENTIFIES_OFFICER_FLAG',
               'VERBAL_IDENTIFIES_OFFICER_FLAG',
               'SUMMONS_ISSUED_FLAG',
               'SUPERVISING_OFFICER_COMMAND_CODE',
               'CPW',
              ]

unusable_cols = ['WHITE',
                 'WHITE HISPANIC',
                 'SUSPECT_SEX',
                 'STOP_LOCATION_X',
                 'STOP_LOCATION_Y',
                 'STOP_LOCATION_PRECINCT',
                 'STATEN ISLAND',
                 'QUEENS',
                 'MONTH2',
                 'MANHATTAN',
                 'DAY2',
                 'BLACK',
                 'BLACK HISPANIC',
                 'BROOKLYN',
                 'ASIAN / PACIFIC ISLANDER'
                ]

# drop unnecessary variables
def drop_unnecessary_variables(df, variables_list):
    return df.drop(variables_list, axis=1)

# dealing with missing values
def deal_with_missing_values(df):
    return df.replace('(null)', np.nan)

# convert Y/N flag columns to boolean
def convert_flags_to_boolean(df):
    binary_map_dict = {'Y': 1, 'N': 0, np.nan: 0, 'I':1, 'S':1, 'V':1}

    # carrying out this operation for flags
    for col in df.columns:
        if 'FLAG' in col:
            df[col] = list(map(lambda x: binary_map_dict[x], df[col]))
    return df

# convert categorical variables to one-hot-encoding
def make_one_hot(df, var):
    one_hot = pd.get_dummies(df[var], drop_first=True)
    df = df.drop(var,axis = 1)
    df = df.join(one_hot)
    return df

# specific transformations for height: converting feet to inches
def replace_feet_inches(h):
    try:
        height = str(h).split('.')
        return int(height[0])*12 + int(height[1])
    except:
        return h

# specific transformations
def specific_variable_transformations(df):
    # height
    df = replace_with_function(df, 'SUSPECT_HEIGHT', replace_feet_inches)
    return df

# replaces values in a variable with a mapping
def replace_with_map(df, var, map_dict):
    df[var] = list(map(lambda x: map_dict[x], df[var]))
    return df

# replaces values in a variable with a function
def replace_with_function(df, var, function):
    df[var] = list(map(lambda x: function(x), df[var]))
    return df

def data_cleaning_pipeline(df):
    # carrying out data cleaning using helper functions defined above
    unnecessary_variables = ['STOP_LOCATION_PREMISES_NAME', 'STOP_FRISK_ID', 'YEAR2', 'RECORD_STATUS_CODE']
    df = drop_unnecessary_variables(df, unnecessary_variables)

    # dealing with missing values
    df = deal_with_missing_values(df)

    # convert Y/N flags to booleans and doing the same for gender
    df = convert_flags_to_boolean(df)
    df = replace_with_map(df, 'SUSPECT_SEX', {'MALE': 1, 'FEMALE': 0, np.nan: np.nan})

    month_list = 'January February March April May June July August September October November December'.split()
    month_dict = {month:i for i, month in enumerate(month_list)}
    df = replace_with_map(df, 'MONTH2', month_dict)

    day_list = 'Monday Tuesday Wednesday Thursday Friday Saturday Sunday'.split()
    day_dict = {month:i for i, month in enumerate(day_list)}
    df = replace_with_map(df, 'DAY2', day_dict)

    stop_dict = {'Based on C/W on Scene': 0, 'Based on Radio Run': 1, 'Based on Self Initiated': 2}
    df = replace_with_map(df, 'STOP_WAS_INITIATED', stop_dict)

    # specific variable transformations
    df = specific_variable_transformations(df)

    # convert categorical variables to one-hot
    df = make_one_hot(df, 'SUSPECT_RACE_DESCRIPTION')
    df = make_one_hot(df, 'SUSPECTED_CRIME_DESCRIPTION')
    df = make_one_hot(df, 'STOP_LOCATION_BORO_NAME')

    # done
    print('Cleaned dataframe shape: {}'.format(df.shape))
    return df

def top_k_features(features, feature_importances, k=10):
    indices = np.argsort(feature_importances)[-k:]
    return indices, features[indices], feature_importances[indices]

def get_X_y(df, drop_cols=[]):
    model_df = df.select_dtypes(include=['int64', 'uint8', 'float64']).dropna(axis=0)
    y = np.array(model_df['SUSPECT_ARRESTED_FLAG'])
    X = model_df[model_df.columns.difference(['SUSPECT_ARRESTED_FLAG', 'SEARCH_BASIS_INCIDENTAL_TO_ARREST_FLAG'])]
    X = X[X.columns.difference(drop_cols)]
    return X, y
