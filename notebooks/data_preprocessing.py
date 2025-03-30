import pandas as pd


def get_structure(bonds_filepath: str, printing = True)-> pd.DataFrame:
    """
    Load the bond index structure data from an Excel file and filter by date.

    This function reads the "structure" sheet from the specified Excel file, converts the 
    "Index Structure Date" column to a datetime format using the '%d.%m.%Y' pattern, and filters 
    out records earlier than January 1, 2018. Optionally, it prints the first few rows of the structure.

    Parameters:
    - bonds_filepath (str): The file path to the Excel file containing bond data.
    - printing (bool): If True, prints the first few rows of the resulting DataFrame (default True).

    Returns:
    - pd.DataFrame: Filtered DataFrame containing the bond index structure data.
    """
    structure = pd.read_excel(
        r"../data/cbonds-usa-corporate-index_upd.xlsx", sheet_name="structure"
    )
    structure["Index Structure Date"] = pd.to_datetime(structure["Index Structure Date"], format="%d.%m.%Y")
    structure = structure[structure["Index Structure Date"] >= pd.Timestamp("2018-01-01")]
    if printing:
        print("Index Structure checking:")
        print(structure.head())
    return structure

def get_quotes(quotes_filepath: str, bonds_filepath: str, printing = True) -> pd.DataFrame:
    """
    Load bond quotes and compute the adjusted ACI percentage.

    This function reads a CSV file containing bond quotes, including "Trade date", "Indicative price, %", 
    "ISIN", and "ACI". It parses the "Trade date" as a datetime object and filters records from January 1, 2018 onward.
    Additionally, it reads bond description data from an Excel file to create a mapping of ISIN to nominal values.
    Using this mapping, it computes a new column "ACI, %" as a percentage adjusted metric and prints the number of missing values.

    Parameters:
    - quotes_filepath (str): File path to the CSV file containing bond quotes.
    - bonds_filepath (str): File path to the Excel file containing bond description data.
    - printing (bool): If True, prints the first few rows of the quotes and the count of missing values (default True).

    Returns:
    - pd.DataFrame: DataFrame with the bond quotes and the computed "ACI, %" column.
    """
    quotes = pd.read_csv(
        quotes_filepath,
        usecols=["Trade date", "Indicative price, %", "ISIN", "ACI"],
        parse_dates=["Trade date"],
    )
    quotes["Trade date"] = pd.to_datetime(quotes["Trade date"], format="ISO8601")
    quotes.drop_duplicates(inplace=True)
    quotes = quotes[quotes["Trade date"] >= pd.Timestamp("2018-01-01")]
    if printing:
        print("Котировки checking:")
        print(quotes.head())

    bonds_desc = pd.read_excel(
        bonds_filepath,
        sheet_name="bonds_desc_15.04.2023",
        usecols=["ISIN", "Coupon frequency", "Nominal / Minimum Settlement Amount"],
    )
    mapping_isin_nominal = dict(zip(bonds_desc["ISIN"], bonds_desc["Nominal / Minimum Settlement Amount"]))
    quotes["ACI, %"] = quotes["ACI"] * 100 / quotes["ISIN"].map(mapping_isin_nominal)
    print("Количество пропусков в ACI, %:", quotes["ACI, %"].isna().sum())
    return quotes


def calc_dirty_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the dirty price for each bond based on its quotes.

    This function creates a copy of the input DataFrame and initializes a "dirty_price" column.
    It processes the DataFrame by setting a multi-index ("ISIN", "Trade date") and sorting the index.
    For each unique ISIN, it fills missing ACI values with 0, detects coupon events by comparing current and 
    next values of "ACI, %", and computes the coupon yield. The cumulative product of (1 + returns) 
    is calculated to recover the price series, which is then adjusted by adding the non-coupon ACI values 
    and forward filled.

    Parameters:
    - df (pd.DataFrame): DataFrame containing bond quotes with columns "ISIN", "Trade date", "Indicative price, %", and "ACI, %".

    Returns:
    - pd.DataFrame: A DataFrame with an added "dirty_price" column representing the computed dirty prices.
    """
    q = df.copy()
    q["dirty_price"] = 0.0
    isins = q["ISIN"].unique()
    q = q.set_index(["ISIN", "Trade date"]).sort_index()

    for isin in isins:
        dt = q.loc[isin].copy()
        dt["ACI, %"].fillna(0, inplace=True)
        coupon_event = dt["ACI, %"] > dt["ACI, %"].shift(-1)
        coupons = dt["ACI, %"][coupon_event] + dt["ACI, %"].shift(-2)[coupon_event]
        coupon_yield = coupons / dt["Indicative price, %"]
        rets = dt["Indicative price, %"].pct_change().fillna(0) + coupon_yield.fillna(0)
        price = (1 + rets).cumprod() * dt["Indicative price, %"].iloc[0]
        price = price.add(dt["ACI, %"][~coupon_event], fill_value=0).ffill()
        q.loc[isin, "dirty_price"] = price.values
    return q.reset_index()


