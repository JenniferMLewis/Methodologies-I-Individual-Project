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
    return df


def wrangle_tb():
    df = tb_narrow(tb_combine(tb_clean(tb_get())))
    return df