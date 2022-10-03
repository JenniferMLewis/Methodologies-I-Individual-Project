import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.api import Holt
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly


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
    Returns Dataset.
    '''
    df = df[df.region == list(df.region[df.estimated_cases == df.estimated_cases.max()])[0]]
    df = df.drop(columns=["region"])
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

# ======== Visualisations ==========

def hist_col(df):
    sns.histplot(data= df, x='estimated_total_pop')
    plt.show()
    sns.histplot(data=df, x= 'estimated_cases')
    plt.show()
    sns.histplot(data=df, x='estimated_new_cases')
    plt.show()
    sns.histplot(data=df, x='estimated_deaths')
    plt.show()
    sns.histplot(data=df, x='pop_to_tb')
    plt.show()

def hist_est(df):
    sns.histplot(data=df, x= 'estimated_cases', color="yellow")
    sns.histplot(data=df, x='estimated_new_cases', color="green")
    sns.histplot(data=df, x='estimated_deaths', color = "red")
    plt.xlabel("Cases (in Millions)")
    plt.title("Existing Cases, New Cases, Deaths (1990-2013)")
    plt.show()

def hist_est2(df):
    sns.histplot(data=df, x= 'estimated_cases', color="yellow")
    sns.histplot(data=df, x='estimated_new_cases', color="green")
    sns.histplot(data=df, x='estimated_deaths', color = "red")
    plt.xlabel("Cases")
    plt.title("Existing Cases, New Cases, Deaths (1990-2013)")
    plt.show()

def barplots(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=df.country, y=df.pop_to_tb, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show()

def boxplots(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x=df.country, y=df.pop_to_tb, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show()

def violin(df):
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.violinplot(x=df.country, y=df.pop_to_tb, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.show()


# ======== For Time Series =========

def wrangle_time():
    df = (tb_combine(tb_clean(tb_get())))
    df = df[df.region == list(df.region[df.estimated_cases == df.estimated_cases.max()])[0]]
    df = df.drop(columns=["region"])
    df = df[df.country == "Myanmar"]
    del df['country']
    df = year_to_dt(tb_difference(pop_to_tb(df)))
    return df

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

def create_y(train):
    '''
    Creates y Dataframe using target
    Returns y
    '''
    y = train.estimated_cases
    return y

def decomp(y):
    '''
    Takes y
    Seasonal decomposes y, then plots the results.
    returns decomposition and decomposition2
    '''
    result = sm.tsa.seasonal_decompose(y)
    decomposition = pd.DataFrame({
        'y': result.observed,
        'trend': result.trend,
        'seasonal': result.seasonal,
        'resid': result.resid,
    })
    print("Cases Seasonality Decomposition:")
    print(decomposition.head())
    print("")
    print("")
    decomposition.iloc[:, 1:].plot()
    decomposition.plot()
    result.plot()
    plt.show()
    return decomposition

def plot_y(y):
    '''
    Takes y
    Plots out the Cases per Year, 
    '''
    ax = y.groupby(y.index.year).mean().plot.bar(width=.9, ec='black')
    plt.xticks(rotation=0)
    ax.set(title='Cases by Year', xlabel='Year', ylabel='Cases')
    plt.show()

def y_tests(train):
    '''
    Takes in train, 
    Creates y [create_y()],
    plots y [plot_y()],
    and decomposes y [decomp()]
    the decomposition aren't required later so it does not return it.
    Returns y
    '''
    y = create_y(train)
    plot_y(y)
    decomp(y)
    return y


def histoplots(df):
    '''
    Takes a Dataframe and plots out the histograms for Myanmar's Estimated cases and deaths. As well as one with both Variables.
    '''
    sns.histplot(data= df, x='estimated_cases',color = 'orange')
    plt.xlabel("Estimated Number of Cases")
    plt.show()
    sns.histplot(data= df, x='estimated_deaths')
    plt.xlabel("Estimated Number of Deaths")
    plt.show()
    sns.histplot(data= df, x='estimated_cases', color = 'orange')
    sns.histplot(data= df, x='estimated_deaths')
    plt.xlabel('Estimated Number of Cases and Deaths')
    plt.show()

def correlation(df):
    '''
    Takes in a dataframe
    Returns the corr, p from stats.pearsonr for estimated_cases, and estimated_deaths.
    '''
    corr, p = stats.pearsonr(df.estimated_cases, df.estimated_deaths)
    print (f'''Correlation : {corr}
    p-value : {p}''')
    df.plot.scatter('estimated_cases', 'estimated_deaths')
    plt.title("Estimaed Cases of TB vs. Estimated Deaths from TB")
    plt.text(250_000, 75_000, f'corr = {corr:.3f}')
    plt.show()

    plt.plot(df.estimated_cases, df.estimated_deaths)
    plt.title('Estimated Cases of TB vs Estimated Deaths from TB')
    plt.xlabel('Estimated Cases')
    plt.ylabel('Estimated Deaths')

    plt.text(250_000, 75_000, f'corr = {corr:.2f}')
    plt.text(290_000, 75_000, f'p = {p:e}')
    plt.show()
    return corr, p

# ========= For Linear Regression =========
# For Later Exploration and Modeling, when time is permitting.

# def df_split(df):
#     train_validate, test = train_test_split(df, test_size=.2, random_state=123)
#     train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

# def df_xy_split(train, validate, test, target="estimated_cases"):
#     '''
#     Takes in train, validate, and test df, as well as target (default: "estimated_cases")
#     Splits them into X, y using target.
#     Returns X, y of train, validate, and test.
#     y sets returned as a proper DataFrame.
#     '''
#     X_train, y_train = train.drop(columns=target), train[target]
#     X_validate, y_validate = validate.drop(columns=target), validate[target]
#     X_test, y_test = test.drop(columns=target), test[target]
#     y_train = pd.DataFrame(y_train)
#     y_validate = pd.DataFrame(y_validate)
#     y_test = pd.DataFrame(y_test)
#     return X_train, y_train, X_validate, y_validate, X_test, y_test

# def scale(df, columns_for_scaling, scaler = MinMaxScaler()):
#     '''
#     Takes in df, columns to be scaled, and scaler (default: MinMaxScaler(); others can be used ie: StandardScaler(), RobustScaler(), QuantileTransformer())
#     Returns a copy of the df, scaled.
#     '''
#     scaled_df = df.copy()
#     scaled_df[columns_for_scaling] = scaler.fit_transform(df[columns_for_scaling])
#     return scaled_df


# ========= Stats Tests =========
alf = 0.05

def stats_start(train):
    y_low = train.estimated_cases[train.index < train.index.mean()]
    y_high = train.estimated_cases[train.index > train.index.mean()]
    return y_low, y_high

def mann(y_low, y_high):
    stat, p = stats.mannwhitneyu(y_low, y_high)
    print('''H0: The mean of estimated cases from 1990 to 1995 is equal to the mean of estimated cases from 1996 to 2001.
Ha: The mean of estimated cases from 1990 to 1995 is not equal to the mean of estimated cases from 1996 to 2001.''')
    print("------")
    if p < alf:
        print("We reject the Null Hypothesis")
    else:
        print("We fail to reject the Null Hypothesis")
    print("")
    print("")
    stat, p = stats.mannwhitneyu(y_low, y_high, alternative='greater')
    print('''H0: The mean of estimated cases from 1990 to 1995 is less than or equal to the mean of estimated cases from 1996 to 2001.
Ha: The mean of estimated cases from 1990 to 1995 is greater than the mean of estimated cases from 1996 to 2001.''')
    print("------")
    if p < alf:
        print("We reject the Null Hypothesis")
    else:
        print("We fail to reject the Null Hypothesis")
    print("")
    print("")
    stat, p = stats.mannwhitneyu(y_low, y_high, alternative='less')
    print('''H0: The mean of estimated cases from 1990 to 1995 is greater than or equal to the mean of estimated cases from 1996 to 2001.
Ha: The mean of estimated cases from 1990 to 1995 is less than to the mean of estimated cases from 1996 to 2001.''')
    print("------")
    if p < alf:
        print("We reject the Null Hypothesis")
    else:
        print("We fail to reject the Null Hypothesis")

def stats_tests(train):
    y_low, y_high = stats_start(train)
    mann(y_low, y_high)

# ======== Modeling =========

def evaluate(validate, yhat_df, target_var):
    '''
    Takes in validate, yhat_df, and target_var
    Uses RMSE to evaluate Validate vs the yhat prediction.
    '''
    rmse = round(sqrt(mean_squared_error(validate.estimated_cases, yhat_df[target_var])), 0)
    return rmse

def plot_and_eval(train, validate, yhat_df ,target_var):
    '''
    Takes Train, Validate, yhat_df, and target_var
    Plots the results of predictions, and Prints the resulting RMSE
    '''
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label = 'Train', linewidth = 1)
    plt.plot(validate[target_var], label = 'Validate', linewidth = 1)
    plt.plot(yhat_df[target_var])
    plt.title(target_var)
    rmse = evaluate(validate, yhat_df, target_var)
    print(target_var, '-- RMSE: {:.0f}'.format(rmse))
    plt.show()

# Create the empty dataframe
def create_eval():
    '''
    Creates a blank eval_df dataframe with columns model_type, target_var, and rmse
    returns the dataframe
    '''
    eval_df = pd.DataFrame(columns=['model_type', 'target_var', 'rmse'])
    return eval_df

# function to store rmse for comparison purposes
def append_eval_df(validate, yhat_df, eval_df, model_type, target_var):
    '''
    Takes in validate, yhat_df, eval_df, model_type, and target_var
    Evalutes the predictions using RMSE
    Creates data to append to the eval_df dataframe.
    Returns the data appended to the eval_df
    '''
    rmse = evaluate(validate, yhat_df, target_var)
    d = {'model_type': [model_type], 'target_var': 'Estimated Cases', 'rmse': [rmse]}
    d = pd.DataFrame(d)
    return eval_df.append(d, ignore_index = True)

def plot_and_eval_test(train, validate, test, yhat_df, target_var):
    '''
    Takes train, validate, test, yhat_df, and target_var
    Plots and Evaluates the results of Evaluating on Test.
    '''
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label = 'Train', linewidth = 1)
    plt.plot(validate[target_var], label = 'Validate', linewidth = 1)
    plt.plot(test[target_var], label = 'Test', linewidth = 1)
    plt.plot(yhat_df[target_var], alpha = .5, color="red")
    plt.title(target_var)
    plt.legend()
    plt.show()

def create_yhat(train, validate):
    '''
    Takes in Train and Validate
    creates a yhat using mean of estimared_cases
    returns yhat_df
    '''
    predicted_cases = train['estimated_cases'][-1:][0]
    yhat_df = pd.DataFrame({'baseline': round(train['estimated_cases'].mean(), 2)}, 
                       index = validate.index)
    return yhat_df

def baseline(train, validate,yhat_df, eval_df):
    yhat_df['estimated_cases'] = round(train['estimated_cases'].mean(), 2)
    eval_df = append_eval_df(validate, yhat_df, eval_df, model_type = 'Baseline', 
                             target_var = "estimated_cases")
    plot_and_eval(train, validate, yhat_df, target_var = "estimated_cases")
    print("predictions:")
    print(yhat_df)
    print("----")
    print(eval_df)
    return eval_df

def last_observed(train, validate,yhat_df, eval_df):
    
    yhat_df['estimated_cases'] = train['estimated_cases'][-1:][0]
    eval_df = append_eval_df(validate, yhat_df, eval_df, model_type = 'Last Observed', 
                             target_var = "estimated_cases")
    plot_and_eval(train, validate, yhat_df, target_var = "estimated_cases")
    print("predictions:")
    print(yhat_df)
    print("----")
    print(eval_df)
    return eval_df


def holts_model(train, validate, yhat_df, eval_df):
    '''
    Takes in train, validate, yhat_df, and eval_df
    Trains to the Holt Prediction Model
    Adds predictions to yhat
    Uses plot_and_eval() to plot out and record evaluations on eval_df
    Returns yhat_df and eval_df
    '''
    model = Holt(train['estimated_cases'], exponential = False)
    model = model.fit(smoothing_level = .1, 
                    smoothing_slope = .1, 
                    optimized = False)
    yhat_cases = model.predict(start = validate.index[0], 
                            end = validate.index[-1])
    yhat_df['estimated_cases'] = round(yhat_cases, 2)
    eval_df = append_eval_df(validate, yhat_df, eval_df, model_type = 'Holts', 
                                target_var = 'estimated_cases')
    plot_and_eval(train, validate, yhat_df, target_var = 'estimated_cases')
    print("predictions:")
    print(yhat_df)
    print("----")
    print(eval_df)
    return eval_df

def fb_prophet(y):
    '''
    Dates in a DataFrame
    Creates New Dataframe using the Data to the specifications in Prophet's documentation,
    Splits the New Dataframe to Train, Validate, and Test
    Fits to Prophet, Creates a Future Dataframe,
    Predicts the Future
    Prints Results and Plots the forecast the model created.
    Returns train, validate, test
    '''
    y = y.copy()
    y = y.reset_index()
    y = y.rename(columns={'year':'ds', 'estimated_cases': 'y'})
    m = Prophet()
    m.fit(y)
    future = m.make_future_dataframe(periods=1)
    future = future[:-1]
    future.tail()
    # f = {'ds': ['1995-02-01', '1995-03-01', '1995-04-01', '1995-05-01', '1995-06-01', '1995-07-01', '1995-0-01', '2009-12-01', '2010-01-01', '2010-02-01', '2010-03-01', '2010-04-01']}
    # f = pd.DataFrame(f)
    # future = future.append(y, ignore_index = True)
    # future.ds = pd.to_datetime(future.ds)
    forecast = m.predict(future)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(15))
    m.plot(forecast)
    plot_plotly(m, forecast)
    plot_components_plotly(m, forecast)


def best_rmse(eval_df, train):
    '''
    Takes in eval_df and train
    finds the lowest rmse,
    prints the lowest rmse as a table
    Bar Plots the comparisions.
    '''
    # get the min rmse for each variable

    min_rmse_cases = eval_df.rmse.min()
    print(eval_df[(eval_df.rmse == min_rmse_cases)])

    x = eval_df['model_type']
    y = eval_df['rmse']
    plt.figure(figsize=(12, 6))
    sns.barplot(x, y)
    plt.title("Predicted Cases")
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.show()


# ======= Test ========


def plot_and_eval_test(train, validate, test, yhat_df, target_var):
    '''
    Takes train, validate, test, yhat_df, and target_var
    Plots and Evaluates the results of Evaluating on Test.
    '''
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label = 'Train', linewidth = 1)
    plt.plot(validate[target_var], label = 'Validate', linewidth = 1)
    plt.plot(test[target_var], label = 'Test', linewidth = 1)
    plt.plot(yhat_df[target_var], alpha = .5, color="red")
    plt.title(target_var)
    plt.legend()
    plt.show()

def test_model(train, validate, test):
    '''
    Takes in train, validate, and test
    Evaluates the Last Observed model on Test
    Prints the Results
    and Plots them.
    '''
    yhat_df = pd.DataFrame({'estimated_cases': validate['estimated_cases'][-1:][0]}, 
                       index = test.index)
    yhat_df.index = test.index
    rmse_cases = round(sqrt(mean_squared_error(test['estimated_cases'], yhat_df['estimated_cases'])), 2)
    print("On Test:")
    print(f"rmse : Estimated Cases: {rmse_cases}")
    plot_and_eval_test(train, validate, test, yhat_df, 'estimated_cases')