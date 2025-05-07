from pandas import read_csv, set_option, DataFrame, concat
from sklearn.preprocessing import OneHotEncoder
from pandas.plotting import scatter_matrix
from seaborn import heatmap, histplot, boxplot
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from numpy import log1p
kickstarter_filename = 'kickstarter_data_full.csv'

ks_data = read_csv(kickstarter_filename, low_memory=False)
print(f'ks_data shape: {ks_data.shape}')
columns_to_drop = ['Unnamed: 0', 'id', 'photo', 'name', 'blurb', 'slug', 'currency_symbol', 'currency_trailing_code', 'static_usd_rate', 'creator', 'profile', 'friends', 'is_backing', 'permissions', 'name_len', 'blurb_len', 'urls', 'source_url', 'location', 'is_starred', 'create_to_launch']
float_columns = ['goal', 'pledged', 'usd_pledged']
int_columns = ['backers_count', 'name_len_clean', 'blurb_len_clean',  'launch_to_deadline', 'launch_to_state_change', 'create_to_launch_days', 'launch_to_deadline_days', 'launch_to_state_change_days', ]
datetime_columns = ['deadline', 'state_changed_at', 'created_at', 'launched_at']
date_int_columns = ['deadline_month', 'deadline_day', 'deadline_hr', 'state_changed_at_month', 'state_changed_at_day', 'state_changed_at_month', 'state_changed_at_day', 'state_changed_at_yr', 'created_at_month', 'created_at_day', 'created_at_hr', 'launched_at_month', 'launched_at_day', 'launched_at_hr', 'state_changed_at_hr', 'created_at_yr', 'launched_at_yr']
category_columns = ['state', 'currency', 'staff_pick', 'category', 'deadline_weekday', 'state_changed_at_weekday', 'created_at_weekday', 'launched_at_weekday', 'deadline_yr', 'country']
boolean_columns = ['disable_communication', 'spotlight', 'SuccessfulBool', 'USorGB', 'TOPCOUNTRY', 'LaunchedTuesday', 'DeadlineWeekend']

kickstarter = ks_data.copy()

kickstarter = kickstarter.drop(columns=columns_to_drop)
kickstarter = kickstarter.dropna(subset=['name_len_clean', 'blurb_len_clean'])
kickstarter = kickstarter.drop(columns=['state', 'pledged', 'usd_pledged'])
float_columns = [col for col in float_columns if col in kickstarter.columns]
category_columns = [col for col in category_columns if col in kickstarter.columns]
kickstarter['name_len_clean'] = kickstarter['name_len_clean'].astype(int)
kickstarter['blurb_len_clean'] = kickstarter['blurb_len_clean'].astype(int)
kickstarter['category'] = kickstarter['category'].fillna("None")

def data_info(_data, show_scatter=False):
    print(f'Data head: {_data.head(5)}')
    print(f'Null values: {_data.isnull().sum()}')
    print(f'Data Shape: {_data.shape[0]} rows and {_data.shape[1]} columns')
    print(f'Columns: {list(_data.columns)}')
    for column in _data.columns:
        if len(_data[column].unique()) < 10:
            print(f'{column} unique values: {_data[column].unique()}')
        else:
            percent_unique = len(_data[column].unique()) / _data.shape[0] * 100
            print(f'{column} % unique values: {percent_unique}')
    print(_data.describe())
    _data.hist(figsize=(12, 10))
    plt.tight_layout()
    plt.show()
    plt.figure()  # new plot
    plt.tight_layout()

    # Only show for floats and ints

    float_columns = _data.select_dtypes(include=['float64']).columns
    int_columns = _data.select_dtypes(include=['int64']).columns
    # Combine float and int columns

    numeric_columns = float_columns.append(int_columns)
    # Calculate correlation matrix
    corMat = _data[numeric_columns].corr(method='pearson')

    print(corMat)
    ## plot correlation matrix as a heat map
    plt.figure(figsize=(14, 10))
    heatmap(corMat, square=True)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.title(f"CORRELATION MATRIX USING HEAT MAP")
    plt.show()

    ## scatter plot of all _data
    plt.figure()
    # # The output overlaps itself, resize it to display better (w padding)
    if show_scatter:
        try:
            scatter_matrix(_data)
            plt.tight_layout(pad=0.1)
            plt.show()
        except:
            return

def data_info2(_date):
    set_option('display.max_columns', None)
    if not isinstance(_date, list):
        if not isinstance(_date, tuple):
            _date = ('', _date)
        _date = [_date]
    for name, data in _date:
        print(f'{name} data')
        print(data.info())
        print(data.head(5))
        for column in data.columns:
            highlight_column = 'profile'
            if column == highlight_column:
                # print 2 rows of values completely
                row1 = data.iloc[0][highlight_column]
                row2 = data.iloc[1][highlight_column]
                print(f'Row 1: {row1}')
                print(f'Row 2: {row2}')
            if len(data[column].unique()) < 10:
                print(f'{column} unique values: {data[column].unique()}')
            else:
                percent_unique = len(data[column].unique()) / data.shape[0] * 100
                print(f'{column} % unique values: {percent_unique}')
        break

def encode_categorical_features(df: DataFrame) -> tuple:
    """Encode all categorical features in the dataframe"""
    # Create a copy to avoid modifying the original
    df = df.copy()

    # Get all categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns: {categorical_columns}")

    boolean_columns = ['staff_pick']
    for col in boolean_columns:
        if col in df.columns:
            # Alternative approach using astype
            df[col] = df[col].map({False: 0, True: 1}).astype(int)
            print(f"Converted boolean column: {col}")

    # Columns to one-hot encode
    columns_to_encode = [
        'category',
        'deadline_weekday',
        'created_at_weekday',
        'launched_at_weekday'
    ]

    # Filter out columns that aren't in the dataframe
    columns_to_encode = [col for col in columns_to_encode if col in df.columns]

    # Remove 'state_changed_at_weekday' column if it exists
    if 'state_changed_at_weekday' in df.columns:
        df = df.drop(columns='state_changed_at_weekday')
        print("Dropped 'state_changed_at_weekday' column")

    # One-hot encode the selected columns
    all_encoded_dfs = []

    for col in columns_to_encode:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(df[[col]])

        # Create meaningful column names
        encoded_cols = [f"{col}_{c}" for c in encoder.categories_[0]]
        encoded_df = DataFrame(encoded, columns=encoded_cols, index=df.index)

        # Add to list of encoded dataframes
        all_encoded_dfs.append(encoded_df)

        # Drop the original column
        print(f'Dropping original column: {col}')
        df = df.drop(columns=col)
        print(f"Encoded column: {col} → {len(encoded_cols)} features")

    # Combine all dataframes
    result = concat([df] + all_encoded_dfs, axis=1)

    # Create a list of all categorical columns (original and encoded)
    all_categorical_columns = []
    for col in categorical_columns:
        if col in result.columns:
            all_categorical_columns.append(col)
        else:
            # Add the encoded column names
            all_categorical_columns.extend([c for c in result.columns if c.startswith(f"{col}_")])

    print(f"Total categorical columns after encoding: {len(all_categorical_columns)}")

    return result, all_categorical_columns

def plot_feature_distribution(data, feature):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    histplot(data[feature], bins=30, kde=True)
    plt.title(f'{feature} Distribution')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    boxplot(x=data[feature])
    plt.title(f'{feature} Boxplot')
    plt.xlabel(feature)

    plt.tight_layout()
    plt.show()

# Example usage
kickstarter, created_category_columns = encode_categorical_features(kickstarter)

columns_to_standardize = [ # Roughly normal
    'name_len_clean', 'blurb_len_clean',
    'deadline_month', 'deadline_day', 'deadline_hr',
    'state_changed_at_month', 'state_changed_at_day', 'state_changed_at_hr',
    'created_at_month', 'created_at_day', 'created_at_hr',
    'launched_at_month', 'launched_at_day', 'launched_at_hr'
]

columns_to_log_transform = [ # Heavily right skewed
    'goal', 'backers_count',
    'create_to_launch_days'
]

columns_to_normalize = [ # Different scales
    'launch_to_deadline_days', 'launch_to_state_change_days'
]

columns_to_drop2 = [
    # Temporal columns represented by better features
    'deadline', 'state_changed_at', 'created_at', 'launched_at',
    'launch_to_deadline', 'launch_to_state_change',

    # Duplicate/redundant information
    'deadline_yr', 'state_changed_at_yr', 'created_at_yr', 'launched_at_yr',

    # Categorical with many unique values, better represented by other features
    'country', 'currency',

    # Low variance or binary columns that might be redundant
    'disable_communication', 'spotlight',
    'USorGB', 'TOPCOUNTRY', 'LaunchedTuesday', 'DeadlineWeekend'
]

# Drop the columns we don't want
kickstarter = kickstarter.drop(columns=columns_to_drop2)
kickstarter = kickstarter.dropna(subset=columns_to_log_transform)
# Standardize the columns

scaler = StandardScaler()
for column in columns_to_standardize:
    kickstarter[column] = scaler.fit_transform(kickstarter[[column]])

# Log transform the columns

log_transformer = FunctionTransformer(log1p, validate=True)
for column in columns_to_log_transform:
    kickstarter[column] = log_transformer.fit_transform(kickstarter[[column]])
# Normalize the columns

normalizer = MinMaxScaler()
for column in columns_to_normalize:
    kickstarter[column] = normalizer.fit_transform(kickstarter[[column]])


from sklearn.model_selection import train_test_split

X = kickstarter.copy()
Y1 = kickstarter['SuccessfulBool']
X = X.drop(columns=['SuccessfulBool', 'backers_count'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, Y1, test_size=0.2, random_state=42)