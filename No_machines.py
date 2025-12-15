import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([tf.keras.layers.Dense(units=2, input_shape=[2])])
#model.compile(optimizer='sgd', loss='mean_squared_error')
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mean_squared_error')

A_rate = 4 # 4 per hour
B_rate = 2 # 2 per hour

Output = np.array([[12,3],[10,3],[6,3],[18,3],[40,4],[28,5],[280,7],[382,4],[28,2],[88,4],[37,3],[3,4],[23,4]], dtype=float) # output in a given time period [amount, hours]
production_rate = (Output[:, 0] / Output[:, 1])
#pro_rate = (19/3)
No_A = np.ceil(production_rate / A_rate).astype(float)  # max number of machines of type A
No_B = np.ceil(production_rate / B_rate).astype(float)  # max number of machines of type B
machines = np.column_stack((No_A, No_B))
print(production_rate, "\n",machines)

model.fit(Output, machines, epochs=2000)
pred = model.predict(np.array([[10,3]], dtype=float))
print(pred)
print(np.round(pred).astype(int))



#No_m = np.array([[1,2],[1,2],[1,1],[2,3]], dtype=int32) # number of machines of each type [A, B]

