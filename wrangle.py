from symbol import tfpdef
import pandas as pd



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
    cols = ['Estimated prevalence of TB (all forms) per 100 000 population, low bound', 'Estimated prevalence of TB (all forms) per 100 000 population, high bound', 
    'Estimated prevalence of TB (all forms), low bound',
    'Estimated prevalence of TB (all forms), high bound',
    'Estimated mortality of TB cases (all forms, excluding HIV), per 100 000 population, low bound',
    'Estimated mortality of TB cases (all forms, excluding HIV), per 100 000 population, high bound',
    'Estimated number of deaths from TB (all forms, excluding HIV), low bound',
    'Estimated number of deaths from TB (all forms, excluding HIV), high bound',
    'Estimated mortality of TB cases who are HIV-positive, per 100 000 population, low bound',
    'Estimated mortality of TB cases who are HIV-positive, per 100 000 population, high bound',
    'Estimated number of deaths from TB in people who are HIV-positive, low bound',
    'Estimated number of deaths from TB in people who are HIV-positive, high bound',
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