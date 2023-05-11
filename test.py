import datasets
X,Y = datasets.load_linear_example1()

print(f'{X=}')
print(f'{Y=}')

import regression
import importlib
importlib.reload(regression)
model = regression.LinearRegression()

#ver2
model.fit(X,Y)
print(model.theta)

#ver3
model.predict(X)
print(model.predict(X))

#ver4
model.score(X,Y)
print(model.score(X,Y))
