import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() # необходимо для работы с tensorflow (отключает особенности функционала второй версии)
x0, x1 = 0, 8 # диапазон аргумента функции

test_data_size = 9 # количество данных для итерации обучения
iterations = 20000 # количество итераций обучения
learn_rate = 0.01 # коэффициент переобучения

hiddenSize = 10 # размер скрытого слоя

curve_x = [] # задаём x для построения кривой (от 0 до 8, с шагом 0.01)
for x in np.arange(0, 8.01, 0.01):
    curve_x.append([x])
    
# функция генерации тестовых величин
def generate_test_values():
    #задаём значения x и y из таблицы
    train_x = [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
    train_y = [[6.45], [4.06], [2.53], [2.05], [2.48], [3.97], [6.57], [9.94], [14.45]]

    return np.array(train_x), np.array(train_y)

# узел на который будем подавать аргументы функции
x = tf.placeholder(tf.float32, [None, 1], name="x")

# узел на который будем подавать значения функции
y = tf.placeholder(tf.float32, [None, 1], name="y")

# скрытый слой
nn = tf.layers.dense(x, hiddenSize,
                     activation=tf.nn.sigmoid, # лог-сигмоидная функция активации
                     kernel_initializer=tf.initializers.ones(), # веса нейронов изначально равны 1
                     bias_initializer=tf.initializers.random_uniform(minval=-x1, maxval=-x0), # инициализация смещения
                     name="hidden")

# выходной слой
model = tf.layers.dense(nn, 1,
                        activation=None,
                        name="output")

# функция подсчёта ошибки
cost = tf.losses.mean_squared_error(y, model)
# минимизация значения функции ошибки методом градиентного спуска
train = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)
# инициализация глобальных переменных среды tensorflow
init = tf.initializers.global_variables()

# запуск сессии tensorflow
with tf.Session() as session:
    session.run(init)
    
    # цикл по итерациям обучения
    for _ in range(iterations):
    
        # данные для обучения (из таблицы)
        train_dataset, train_values = generate_test_values()
        # загружаем обучающие данные в нейросеть
        session.run(train, feed_dict={
            x: train_dataset,
            y: train_values
        })
        # Выводим каждые 1000 итерций значение фукнции подсчёта ошибки
        if(_ % 1000 == 999):
            print("cost = {}".format(session.run(cost, feed_dict={
                x: train_dataset,
                y: train_values
            })))
    
    # на основе обученной модели получаем значение y
    curve_y = session.run(model, feed_dict={
        x: curve_x,
    })
    
    # строим графики (кривая красным цветом, синим помечены исходные точки)
    plt.plot(curve_x, curve_y, "r", train_dataset, train_values, "bo")
    plt.show()
    # выводим значения весов и смещений нейронов на скрытом слое
    with tf.variable_scope("hidden", reuse=True):
        w = tf.get_variable("kernel")
        b = tf.get_variable("bias")
        print("hidden:")
        print("kernel=", w.eval())
        print("bias = ", b.eval())
    # выводим значения весов и смещений нейронов на выходном слое
    with tf.variable_scope("output", reuse=True):
        w = tf.get_variable("kernel")
        b = tf.get_variable("bias")
        print("output:")
        print("kernel=", w.eval())
        print("bias = ", b.eval())