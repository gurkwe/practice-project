from sklearn.ensemble import RandomForestRegressor
from useful_package.module_a import polynom_3
from useful_package.module_b import hyperbola

def calc_MSE(X_train, Y_train, X_test, Y_test):
    forest =  RandomForestRegressor()
    forest.fit(X_train, Y_train)
    pred = forest.predict(X_test)
    n = len(Y_test)
    mse = sum((Y_test - pred)**2)/n
    return mse

X = []
for i in range(1,10):
    X.append([i])

Ya = [polynom_3(x[0]) for x in X]
Yb = [hyperbola(x[0]) for x in X]

n = 5

mse_a = calc_MSE(X[:n], Ya[:n], X[n:], Ya[n:])
mse_b = calc_MSE(X[:n], Yb[:n], X[n:], Yb[n:])

print("MSE A:", mse_a)
print("MSE B:", mse_b)
