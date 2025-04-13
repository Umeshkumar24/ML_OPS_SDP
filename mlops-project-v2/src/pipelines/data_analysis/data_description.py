def describe_and_analyze_data(data):
    """
    Perform data description and analysis.
    """
    print("\n--- Data Description ---")
    print(data.describe())

    print("\n--- Data Information ---")
    print(data.info())

    print("\n--- Checking for Missing Values ---")
    print(data.isnull().sum())

    print("\n--- Correlation Matrix ---")
    print(data.corr())
