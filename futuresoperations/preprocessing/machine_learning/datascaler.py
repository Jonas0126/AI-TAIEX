from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Datascaler:
    def __init__(self):
        self.stdscalers = {}
        self.pca = None

    def scaling(self, numpy_traindata, numpy_testdata, dopca):
        x_train = numpy_traindata[0]
        x_test = numpy_testdata[0]
        x_train, x_test = self.standard_scaling(x_train, x_test)
        if dopca is True:
            x_train, x_test = self.pca_transform(x_train, x_test)
        train_data = (x_train, numpy_traindata[1])
        test_data = (x_test, numpy_testdata[1])
        return train_data, test_data
    
    def transform(self, data, dopca):
        transformed_data = data
        for col_num in range(data.shape[1]):
            data_col = data[:, col_num].reshape(-1, 1)
            transformed_data[:, col_num] = self.stdscalers[col_num].transform(data_col).squeeze()
        if dopca is True:
            transformed_data = self.pca.transform(transformed_data)
        return transformed_data

    def standard_scaling(self, x_train, x_test):
        stocks_train = x_train
        stocks_test = x_test
        for col_num in range(x_train.shape[1]):
            stdscaler = StandardScaler()
            x_train_col = x_train[:, col_num].reshape(-1, 1)
            x_test_col = x_test[:, col_num].reshape(-1, 1)
            stocks_train[:, col_num] = stdscaler.fit_transform(x_train_col).squeeze()
            stocks_test[:, col_num] = stdscaler.transform(x_test_col).squeeze()
            self.stdscalers[col_num] = stdscaler
        return stocks_train, stocks_test
    
    def pca_transform(self, x_train, x_test):
        self.pca = PCA(n_components=0.95)
        stocks_train = self.pca.fit_transform(x_train)
        stocks_test = self.pca.transform(x_test)
        return stocks_train, stocks_test