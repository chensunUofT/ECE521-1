from numpy import *
from pylab import *
from matplotlib.pyplot import *

def gradientDescentStep(x, y, theta, alpha):
    y_model = x.dot(theta).flatten()
    error = (y_model - y.flatten())
    m=y.size
    theta[0][0] -= alpha * (1.0/m) * (error).sum();
    theta[1][0] -= alpha * (1.0/m) * (error*x[:,1]).sum();
    return theta

# Generate artificial dataset
train_x = linspace(1.0, 10.0, num=100)[:, newaxis]
train_y = sin(train_x) + 0.1 * power(train_x, 2) + 0.5 * random.randn(100, 1)
tSize = train_x.size

# Precalculate input powers
xp = zeros(shape=(tSize,2))
for i in range(2):
    xp[:,i] = power(train_x, i)[:,0]

# Linear regression parameters
iterations = 2000
lRate = 0.001
theta = zeros(shape=(2,1))


for i in range(iterations):
    theta = gradientDescentStep(xp, train_y, theta, lRate)
    if i%1000 == 0:
        print(i, theta)

print(theta)
scatter(train_x, train_y, marker='x', color='b')
result = xp.dot(theta).flatten()
plot(train_x, result)
show()





