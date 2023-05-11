import datasets
X,Y = datasets.load_linear_example1()

print(f'{X=}')
print(f'{Y=}')

import regression
model = regression.LinearRegression()
model.x