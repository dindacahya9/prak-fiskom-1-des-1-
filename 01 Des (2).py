import numpy as np
from sklearn.linear_model import LinearRegression

#Database
# x = Data, y = Target
x = [[1],[3],[5],[7],[9],[11],[13],[15],[17],[19]]
y = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]

regr = LinearRegression().fit(x,y)
regr.score(x,y)



#Data uji
predict = np.array([[15]])

#Menampilkan data prediksi
print ("Prediksi")
print ("Input = ", predict)
print ("Output = ", regr.predict(predict))
