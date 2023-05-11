import datasets
X,Y = datasets.load_linear_example1()

print(f'{X=}')
print(f'{Y=}')

import regression
import importlib
importlib.reload(regression)
model = regression.LinearRegression()
model.fit(X,Y)
print(model.theta)
