from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Datascaler:
    def __init__(self):
        self.stdscalers = {}
        self.pca = None

    def scaling(self, train_features, dopca):
        train_features = self.standard_scaling(train_features)
        if dopca is True:
            _ = self.pca_transform(train_features)
    
    def transform(self, data, dopca):
        transformed_data = data
        features_name = data.columns
        for feature_name in features_name:
            stdscaler = self.stdscalers[feature_name]
            feature_data = data[feature_name].to_numpy().reshape(-1, 1)
            transformed_feature_data = stdscaler.transform(feature_data)
            transformed_data[feature_name] = transformed_feature_data
        if dopca is True:
            transformed_data = self.pca.transform(transformed_data)
        return transformed_data

    def standard_scaling(self, train_features):
        transformed_train_features = train_features
        features_name = train_features.columns
        for feature_name in features_name:
            x_train = train_features[feature_name].to_numpy().reshape(-1, 1)
            stdscaler = StandardScaler()
            transformed_x_train = stdscaler.fit_transform(x_train)
            transformed_train_features[feature_name] = transformed_x_train
            self.stdscalers[feature_name] = stdscaler
        return transformed_train_features

    def pca_transform(self, train_features):
        self.pca = PCA(n_components=0.95)
        stocks_train = self.pca.fit_transform(train_features)
        return stocks_train