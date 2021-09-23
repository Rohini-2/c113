import pandas as pd
import plotly.express as px

df = pd.read_csv("velocity.csv")

velocity_list = df["Velocity"].tolist()
melted_list = df["Escaped"].tolist()

fig = px.scatter(x=velocity_list, y=melted_list)
fig.show()


import numpy as np
velocity_array = np.array(velocity_list)
melted_array = np.array(melted_list)

#Slope and intercept using pre-built function of Numpy
m, c = np.polyfit(velocity_array, melted_array, 1)

y = []
for x in velocity_array:
  y_value = m*x + c
  y.append(y_value)

#plotting the graph
fig = px.scatter(x=velocity_array, y=melted_array)

fig.update_layout(shapes=[
    dict(
      type= 'line',
      y0= min(y), y1= max(y),
      x0= min(velocity_array), x1= max(velocity_array)
    )
])
fig.show()


import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
X=np.reshape(velocity_list,(len(velocity_list),1))
print(X.ravel())
Y=np.reshape(melted_list,(len(melted_list),1))
lr=LogisticRegression()
lr.fit(X,Y)
plt.scatter(X.ravel(),Y.ravel(),color="black")

def model(x):
  return 1/(1+np.exp(-x))

X_test=np.linspace(0,5000,10000)
melting_chances=model(X_test*lr.coef_+lr.intercept_).ravel()  
plt.plot(X_test,melting_chances,color="cyan",linewidth=3)
plt.axhline(y=0,color="blue",linestyle='-')
plt.axhline(y=1,color="blue",linestyle='-')
plt.axhline(y=0.5,color="blue",linestyle='--')
plt.axvline(x=X_test[6843],color="purple",linestyle='--')
plt.xlim(3400,3450)
plt.show()


velocity = float(input("Enter the velocity here:- "))
chances = model(velocity * lr.coef_ + lr.intercept_).ravel()[0]
if chances <=0.01:
  print("rocket will not launch")
elif chances >= 1:
  print("rokect will launch")
elif chances < 0.5:
  print("roket will not launch")
else:
  print("may be rocket get launched")