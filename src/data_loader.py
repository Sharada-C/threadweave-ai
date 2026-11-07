"""
ThreadWeave AI - Data loading utilities
"""
import os
import pandas as pd

from . import config


def load_sales_history() -> pd.DataFrame:
    df = pd.read_csv(config.SALES_HISTORY_CSV, parse_dates=["Date"])
    return df


def load_supplier_data() -> pd.DataFrame:
    return pd.read_csv(config.SUPPLIER_DATA_CSV)


def load_inventory() -> pd.DataFrame:
    return pd.read_csv(config.INVENTORY_CSV)


def load_product_catalog() -> pd.DataFrame:
    return pd.read_csv(config.PRODUCT_CATALOG_CSV)


def output_exists(agent_name: str) -> bool:
    path = config.AGENT_OUTPUT_FILES[agent_name]
    return os.path.exists(path) and os.path.getsize(path) > 0


def load_output(agent_name: str) -> pd.DataFrame:
    path = config.AGENT_OUTPUT_FILES[agent_name]
    return pd.read_csv(path)


def filter_products(df: pd.DataFrame, product_filter: list, id_col: str = "Product_ID",
                     name_col: str = "Product_Name") -> pd.DataFrame:
    """Filter a dataframe down to specific products by ID or (partial, case
    insensitive) name match. Returns the original df if product_filter is
    empty/None."""
    if not product_filter:
        return df
    mask = df[id_col].isin(product_filter)
    if name_col in df.columns:
        name_mask = df[name_col].str.lower().apply(
            lambda name: any(pf.lower() in name for pf in product_filter)
        )
        mask = mask | name_mask
    return df[mask]
