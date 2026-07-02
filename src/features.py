from sklearn.base import BaseEstimator, TransformerMixin

class FlightFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.carrier_stats = None
        self.origin_monthly_stats = None
        self.route_stats = None
        self.global_mean = None

    def fit(self, X, y):
        temp = X.copy()
        temp["IS_DELAYED"] = y.values

        self.carrier_stats = temp.groupby("AIRLINE")["IS_DELAYED"].mean()
        self.origin_monthly_stats = temp.groupby(["ORIGIN_AIRPORT", "MONTH"])["IS_DELAYED"].mean()
        self.route_stats = temp.groupby(["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"])["IS_DELAYED"].mean()
        self.global_mean = y.mean()
        return self

    def transform(self, X):
    X = X.copy()
    
    X['SCHED_DEP_HOUR'] = (X['SCHEDULED_DEPARTURE'] // 100) % 24
    X['IS_PEAK_SEASON'] = X['MONTH'].isin([6, 7, 12]).astype(int)
    X['IS_LATE_NIGHT'] = X['SCHED_DEP_HOUR'].isin([22, 23, 0]).astype(int)
    X['IS_EARLY_MORNING'] = X['SCHED_DEP_HOUR'].isin([5, 6, 7]).astype(int)

    X['CARRIER_DELAY_RATE'] = X['AIRLINE'].map(self.carrier_stats)
    X['ORIGIN_MONTHLY_DELAY_RATE'] = X.set_index(['ORIGIN_AIRPORT', 'MONTH']).index.map(self.origin_monthly_stats).values
    X['ROUTE_DELAY_RATE'] = X.set_index(['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']).index.map(self.route_stats).values
    
    X['CARRIER_DELAY_RATE'] = X['CARRIER_DELAY_RATE'].fillna(self.global_mean)
    X['ORIGIN_MONTHLY_DELAY_RATE'] = X['ORIGIN_MONTHLY_DELAY_RATE'].fillna(self.global_mean)
    X['ROUTE_DELAY_RATE'] = X['ROUTE_DELAY_RATE'].fillna(self.global_mean)
    
    return X
