import pandas as pd


def get_structure(bonds_filepath, printing = True):
    structure = pd.read_excel(
        r"../data/cbonds-usa-corporate-index_upd.xlsx", sheet_name="structure"
    )
    structure["Index Structure Date"] = pd.to_datetime(structure["Index Structure Date"], format="%d.%m.%Y")
    structure = structure[structure["Index Structure Date"] >= pd.Timestamp("2018-01-01")]
    if printing:
        print("Index Structure checking:")
        print(structure.head())
    return structure

def get_quotes(quotes_filepath, bonds_filepath, printing = True):
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


