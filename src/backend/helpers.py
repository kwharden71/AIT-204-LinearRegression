import pandas as pd

def train_test_split(df: pd.DataFrame, percentage: int) -> tuple[pd.DataFrame, pd.DataFrame]:   
    """
    Splits the DataFrame into training and testing sets based on the given percentage.

    Args:
        df: The input DataFrame to be split.
        percentage: The percentage of data to be used for training (0-100).

    Returns:
        A tuple containing the training DataFrame and testing DataFrame.
    """
    if not (0 < percentage < 100):
        raise ValueError("Percentage must be between 0 and 100.")

    train_size = int(len(df) * (percentage / 100))
    train_df = df.iloc[:train_size].reset_index(drop=True)
    test_df = df.iloc[train_size:].reset_index(drop=True)

    return train_df, test_df