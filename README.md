# AutoML

PyTorch AutoML을 몇 가지 모델로 테스트 해 볼 예정


</br></br>
* * *
## AutoViz를 통한 빠른 데이터 시각화
- [공식 git link](https://github.com/AutoViML/AutoViz)
- [pip install link](https://pypi.org/project/autoviz/)
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






