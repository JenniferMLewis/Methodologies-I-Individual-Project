import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer


def tb_get():
    '''
    Loads the TB burden by country data set from the "tb_burden_country.csv"
    Returns the data as a pandas dataframe.
    '''
    df = pd.read_csv("tb_burden_country.csv")
    return df

def tb_clean(df):
    '''
    Takes in the dataframe and removes all low and high bound columns, as well as mainly null ones.
    Returns the now more managable dataframe
    '''
    cols = ['ISO 2-character country/territory code',
    'ISO 3-character country/territory code',
    'ISO numeric country/territory code',
    'Estimated prevalence of TB (all forms) per 100 000 population, low bound', 'Estimated prevalence of TB (all forms) per 100 000 population, high bound', 
    'Estimated prevalence of TB (all forms), low bound',
    'Estimated prevalence of TB (all forms), high bound',
    'Method to derive prevalence estimates',
    'Estimated mortality of TB cases (all forms, excluding HIV), per 100 000 population, low bound',
    'Estimated mortality of TB cases (all forms, excluding HIV), per 100 000 population, high bound',
    'Estimated number of deaths from TB (all forms, excluding HIV), low bound',
    'Estimated number of deaths from TB (all forms, excluding HIV), high bound',
    'Estimated mortality of TB cases who are HIV-positive, per 100 000 population, low bound',
    'Estimated mortality of TB cases who are HIV-positive, per 100 000 population, high bound',
    'Estimated number of deaths from TB in people who are HIV-positive, low bound',
    'Estimated number of deaths from TB in people who are HIV-positive, high bound',
    'Method to derive mortality estimates',
    'Estimated incidence (all forms) per 100 000 population, low bound',
    'Estimated incidence (all forms) per 100 000 population, high bound',
    'Estimated number of incident cases (all forms), low bound',
    'Estimated number of incident cases (all forms), high bound',
    'Method to derive incidence estimates',
    'Case detection rate (all forms), percent',
    'Case detection rate (all forms), percent, low bound',
    'Case detection rate (all forms), percent, high bound',
    'Method to derive TBHIV estimates',
    'Estimated incidence of TB cases who are HIV-positive, high bound',
    'Estimated incidence of TB cases who are HIV-positive, low bound',
    'Estimated incidence of TB cases who are HIV-positive',
    'Estimated HIV in incident TB (percent)',
    'Estimated HIV in incident TB (percent), low bound',
    'Estimated HIV in incident TB (percent), high bound',
    'Estimated incidence of TB cases who are HIV-positive per 100 000 population',
    'Estimated incidence of TB cases who are HIV-positive per 100 000 population, low bound',
    'Estimated incidence of TB cases who are HIV-positive per 100 000 population, high bound']
    for col in cols:
        del df[col]
    return df

def tb_combine(df):
    '''
    Takes in the Dataframe
    Combines columns seperated by HIV-Positive and HIV-Negative,
    Drops the seperated columns,
    Renames all the columns into a more Python friendly manner,
    Returns the Dataframe
    '''
    # Okay, I'm not focusing on HIV, so now that I've confirmed which columns HIV is and isn't included in, let's get to creating just normal columns!
    df["estimated_mortality_per_100k"] = df["Estimated mortality of TB cases (all forms, excluding HIV) per 100 000 population"] + df["Estimated mortality of TB cases who are HIV-positive, per 100 000 population"]
    df["estimated_deaths"] = df["Estimated number of deaths from TB (all forms, excluding HIV)"] + df["Estimated number of deaths from TB in people who are HIV-positive"]
    df = df.drop(columns=["Estimated mortality of TB cases (all forms, excluding HIV) per 100 000 population", "Estimated mortality of TB cases who are HIV-positive, per 100 000 population", "Estimated number of deaths from TB (all forms, excluding HIV)", "Estimated number of deaths from TB in people who are HIV-positive"])
    df = df.rename(columns={"Estimated number of incident cases (all forms)": "estimated_new_cases", 'Estimated incidence (all forms) per 100 000 population' : "estimated_new_case_per_100k", "Estimated prevalence of TB (all forms)": "estimated_cases", "Estimated prevalence of TB (all forms) per 100 000 population" : "estimated_cases_per_100k", "Estimated total population number" : "estimated_total_pop", "Year":"year", "Region": "region", "Country or territory name": "country"})
    return df

def tb_narrow(df):
    '''
    Takes in the Dataframe,
    Finds the Region with the Highest Estimated Cases (SEA for this Dataset),
    '''
    df = df[df.region == list(df.region[df.estimated_cases == df.estimated_cases.max()])[0]]
    df = df.drop(columns=["region"])
    df = df[df.country == "Myanmar"]
    del df['country']
    return df

def pop_to_tb(df):
    '''
    Takes the Dataframe, 
    Calculates how many people to 1 case of TB.
     [So the Lower the Number, the More Prevelant TB is.
     ex. 1 case for every 57 people vs. 1 case for every 202 people. 57 is the higher Prevelance.]
    Returns the Dataframe with a new column of how many people to 1 case.
     (Yes, Cases per 100k will give you the same result of prevelance, but the 1 in x is easier for me to digest, and really shows how common it might be.)
    '''
    df['pop_to_tb'] = round(df.estimated_total_pop / df.estimated_cases)
    return df

def tb_difference(df):
    '''
    Takes in the DataFrame,
    Calculates the difference in current cases, new cases, and deaths between the current year and the previous recorded year, by region.
    Returns the Dataframe with the three new columns for difference.
    Fills the first row with 0 since there is no previous row to compare with, it's the baseline at 0.
    '''
    est_diff = df.estimated_cases.diff()
    death_diff = df.estimated_deaths.diff()
    new_diff = df.estimated_new_cases.diff()
    df_est_diff = df.assign(case_difference = est_diff)
    new_df = df_est_diff.assign(death_difference = death_diff)
    df = new_df.assign(new_case_difference = new_diff)
    df['case_difference'] = df.case_difference.fillna(0)
    df['death_difference'] = df.death_difference.fillna(0)
    df['new_case_difference'] = df.new_case_difference.fillna(0)
    return df


def wrangle_tb():
    df = tb_difference(pop_to_tb(tb_narrow(tb_combine(tb_clean(tb_get())))))
    return df


# ======== For Time Series =========

def year_to_dt(df):
    df['year'] = pd.to_datetime(df['year'], format='%Y')
    df = df.set_index('year').sort_index()
    return df

def time_split(df):
    '''
    Splits the Dataframe into 50%, 30% and 20% then plots them to show no gaps or overlapping
    Returns the splits as Train, Validate and Test.
    '''
    train_size = int(len(df) * .5)
    validate_size = int(len(df) * .3)
    test_size = int(len(df) - train_size - validate_size)
    validate_end_index = train_size + validate_size

    # split into train, validation, test
    train = df[: train_size]
    validate = df[train_size : validate_end_index]
    test = df[validate_end_index : ]
    for col in train.columns:
        plt.figure(figsize=(12,4))
        plt.plot(train[col])
        plt.plot(validate[col])
        plt.plot(test[col])
        plt.ylabel(col)
        plt.show()
    return train, validate, test


# ========= For Linear Regression =========

def df_split(df):
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

def df_xy_split(train, validate, test, target="estimated_cases"):
    '''
    Takes in train, validate, and test df, as well as target (default: "estimated_cases")
    Splits them into X, y using target.
    Returns X, y of train, validate, and test.
    y sets returned as a proper DataFrame.
    '''
    X_train, y_train = train.drop(columns=target), train[target]
    X_validate, y_validate = validate.drop(columns=target), validate[target]
    X_test, y_test = test.drop(columns=target), test[target]
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)
    return X_train, y_train, X_validate, y_validate, X_test, y_test

def scale(df, columns_for_scaling, scaler = MinMaxScaler()):
    '''
    Takes in df, columns to be scaled, and scaler (default: MinMaxScaler(); others can be used ie: StandardScaler(), RobustScaler(), QuantileTransformer())
    Returns a copy of the df, scaled.
    '''
    scaled_df = df.copy()
    scaled_df[columns_for_scaling] = scaler.fit_transform(df[columns_for_scaling])
    return scaled_df