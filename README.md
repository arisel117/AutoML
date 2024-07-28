# AutoML

PyTorch AutoML을 몇 가지 모델로 테스트 해 볼 예정


</br></br>
* * *
## AutoViz를 통한 빠른 데이터 시각화
- [공식 git link](https://github.com/AutoViML/AutoViz)
- [Install link](https://pypi.org/project/autoviz/)
- 간단 사용 방법
  ```python
  from autoviz.AutoViz_Class import AutoViz_Class

  df = pd.read_csv('my_dataset.csv')

  AV = AutoViz_Class()
  AV.AutoViz(
      filename='', 
      sep=',', 
      depVar='', 
      dfte=df, 
      header=0, 
      verbose=0, 
      lowess=False, 
      chart_format='svg', 
      max_rows_analyzed=10000, 
      max_cols_analyzed=30,
      save_plot_dir='./result'
  )
  ```
* * *

</br></br>

* * *
## lazypredict를 통해 빠르게 학습 모델 구축 및 테스트
- 간단하게 모델을 다양한 모델로 빠르게 돌려볼 수 있는 AutoML
- [공식 git link](https://github.com/shankarpandala/lazypredict)
- [Install link](https://pypi.org/project/lazypredict/)
- 간단 사용 방법
  - Regressor Model
    ```python
    from lazypredict.Supervised import LazyRegressor

    reg = LazyRegressor(verbose=0, predictions=True)

    models, predictions = reg.fit(x_train, x_test, y_train, y_test)
    ```
  - Classifier Model
    ```python
    from lazypredict.Supervised import LazyClassifier

    clf = LazyClassifier(verbose=0, predictions=True)

    models, predictions = clf.fit(x_train, x_test, y_train, y_test)
    ```
* * *

</br></br>

* * *
## auto-sklearn를 통해 빠르게 학습 모델 구축 및 테스트
- 간단하게 모델을 다양한 모델로 빠르게 돌려볼 수 있는 AutoML
- [공식 git link](https://automl.github.io/auto-sklearn/master/#)
- [Install link](https://pypi.org/project/auto-sklearn/)
- 간단 사용 방법
  - Regressor Model
    ```python
    from pprint import pprint

    import sklearn.datasets
    import sklearn.metrics
    
    import autosklearn.regression
    import matplotlib.pyplot as plt

    automl = AutoSklearnRegressor()
    automl.fit(X_train, y_train)

    print(automl.leaderboard())

    pprint(automl.show_models(), indent=4)

    train_predictions = automl.predict(X_train)
    print("Train R2 score:", sklearn.metrics.r2_score(y_train, train_predictions))
    test_predictions = automl.predict(X_test)
    print("Test R2 score:", sklearn.metrics.r2_score(y_test, test_predictions))

    plt.scatter(train_predictions, y_train, label="Train samples", c="#d95f02")
    plt.scatter(test_predictions, y_test, label="Test samples", c="#7570b3")
    plt.xlabel("Predicted value")
    plt.ylabel("True value")
    plt.legend()
    plt.plot([30, 400], [30, 400], c="k", zorder=0)
    plt.xlim([30, 400])
    plt.ylim([30, 400])
    plt.tight_layout()
    plt.show()
    ```
  - Classifier Model
    ```python
    import numpy as np
    from pprint import pprint
    
    import sklearn.datasets
    import sklearn.metrics
    
    import autosklearn.classification

    automl = autosklearn.classification.AutoSklearnClassifier()
    automl.fit(X_train, y_train)

    print(automl.leaderboard())

    pprint(automl.show_models(), indent=4)

    print(automl.sprint_statistics())

    predictions = automl.predict(X_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))

    ```
* * *

</br></br>

* * *
## Auto-PyTorch를 통해 빠르게 학습 모델 구축 및 테스트
- 간단하게 모델을 다양한 모델로 빠르게 돌려볼 수 있는 PyTorch AutoML
- [공식 git link](https://github.com/automl/Auto-PyTorch)
- [Install link](https://pypi.org/project/autoPyTorch/)
- 간단 사용 방법
  - Regressor Model
    ```python
    from autoPyTorch.api.time_series_forecasting import TimeSeriesForecastingTask

    # data and metric imports
    from sktime.datasets import load_longley
    targets, features = load_longley()

    # define the forecasting horizon
    forecasting_horizon = 3

    # Dataset optimized by APT-TS can be a list of np.ndarray/ pd.DataFrame where each series represents an element in the 
    # list, or a single pd.DataFrame that records the series
    # index information: to which series the timestep belongs? This id can be stored as the DataFrame's index or a separate
    # column
    # Within each series, we take the last forecasting_horizon as test targets. The items before that as training targets
    # Normally the value to be forecasted should follow the training sets
    y_train = [targets[: -forecasting_horizon]]
    y_test = [targets[-forecasting_horizon:]]

    # same for features. For uni-variant models, X_train, X_test can be omitted and set as None
    X_train = [features[: -forecasting_horizon]]
    # Here x_test indicates the 'known future features': they are the features known previously, features that are unknown
    # could be replaced with NAN or zeros (which will not be used by our networks). If no feature is known beforehand,
    # we could also omit X_test
    known_future_features = list(features.columns)
    X_test = [features[-forecasting_horizon:]]

    start_times = [targets.index.to_timestamp()[0]]
    freq = '1Y'

    # initialise Auto-PyTorch api
    api = TimeSeriesForecastingTask()

    # Search for an ensemble of machine learning algorithms
    api.search(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test, 
        optimize_metric='mean_MAPE_forecasting',
        n_prediction_steps=forecasting_horizon,
        memory_limit=16 * 1024,  # Currently, forecasting models use much more memories
        freq=freq,
        start_times=start_times,
        func_eval_time_limit_secs=50,
        total_walltime_limit=60,
        min_num_test_instances=1000,  # proxy validation sets. This only works for the tasks with more than 1000 series
        known_future_features=known_future_features,
    )

    # our dataset could directly generate sequences for new datasets
    test_sets = api.dataset.generate_test_seqs()

    # Calculate test accuracy
    y_pred = api.predict(test_sets)
    score = api.score(y_pred, y_test)
    print("Forecasting score", score)
    ```
  - Classifier Model
    ```python
    from autoPyTorch.api.tabular_classification import TabularClassificationTask

    # data and metric imports
    import sklearn.model_selection
    import sklearn.datasets
    import sklearn.metrics
    X, y = sklearn.datasets.load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(X, y, random_state=1)

    # initialise Auto-PyTorch api
    api = TabularClassificationTask()

    # Search for an ensemble of machine learning algorithms
    api.search(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        optimize_metric='accuracy',
        total_walltime_limit=300,
        func_eval_time_limit_secs=50
    )

    # Calculate test accuracy
    y_pred = api.predict(X_test)
    score = api.score(y_pred, y_test)
    print("Accuracy score", score)
    ```
  
* * *

</br></br>

* * *
## 학습 과정 시각화 방법
- visdom를 통해 빠르게 학습 과정 시각화
  - [공식 git link](https://github.com/fossasia/visdom)
  - [Install link](https://pypi.org/project/visdom/)
- tensorboard 이용
  - [tensorflow로 사용하기](https://www.tensorflow.org/tensorboard/get_started?hl=ko)
  - [pytorch로 사용하기](https://tutorials.pytorch.kr/recipes/recipes/tensorboard_with_pytorch.html)
  - [공식 git link](https://github.com/tensorflow/tensorboard)
  - [Install link](https://pypi.org/project/tensorboard/)
- 
* * *
