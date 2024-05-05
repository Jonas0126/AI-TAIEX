# Task Examples
In the example directory, we provide the example tasks with two different targets. One is for S&P 500, the other is for Taiex.

In the target directory, there are many example files and each of them is a task for training models and simulating trading a whole year.

## Configuration
The configurations in the example files are described as below.
* train dates
* test dates
* trade dates
* model
* target name
* futures name
* futures rule
  * set the trading rules of the futures
  * must include fixed months, total front month, delivery week, and delivery weekday
* predict range
  * predict the next few months
  * for example, 2 means it can predict the next month and the second month
* features
  * 'Yahoo Finance ticker': 'feature name'
* hyperparameters
  * only lag can be set
  * lag is called lookback here
* preprocessing
  * correlations: null and number
  * scaling: true and false
  * pca: true and false
* trading
  * for trading simulation
