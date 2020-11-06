import sys
import logging
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class NaiveBayesGaussian:
    """
    Classe do classificador NaiveBayesGaussian
    """

    def __init__(self):
        self.classes = None
        self.class_freq = None
        self.means = {}
        self.std = {}
        self.class_prob = None

    def separate_by_classes(self, x, y):
        """
        Segmenta as entradas por classe para futuramente calcular as médias e desvio padrão
            :param self: instância da classe
            :param x: lista de entradas
            :param y: lista de classes
            :return: dataset segmentado por classes
        """
        y = y.to_numpy()
        x = x.to_numpy()
        self.classes = np.unique(y)
        subdatasets = {}
        classes_index = {}
        cls, counts = np.unique(y, return_counts=True)
        self.class_freq = dict(zip(cls, counts))

        logging.debug(self.class_freq)

        for class_type in self.classes:
            classes_index[class_type] = np.argwhere(y == class_type)
            subdatasets[class_type] = x[classes_index[class_type], :]
            self.class_freq[class_type] = self.class_freq[class_type] / sum(list(self.class_freq.values()))
        return subdatasets

    def fit(self, x, y):
        """
        Realiza o treinamento do modelo
            :param self: instância da classe
            :param x: lista de entradas
            :param y: lista de classes
        """
        separated_X = self.separate_by_classes(x, y)
        for class_type in self.classes:
            self.means[class_type] = np.mean(separated_X[class_type], axis=0)[0]
            self.std[class_type] = np.std(separated_X[class_type], axis=0)[0]

    def calculate_probability(self, x, mean, stdev):
        """
        Realiza o cálculo das probabilidades
            :param self: instância da classe
            :param x: lista de entradas
            :param mean: médias
            :param stdev: desvio padrão
            :return: probabilidades
        """
        exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def predict_proba(self, x):
        self.class_prob = {cls: math.log(self.class_freq[cls], math.e) for cls in self.classes}
        for cls in self.classes:
            for i in range(len(self.means)):
                self.class_prob[cls] += math.log(self.calculate_probability(x[i], self.means[cls][i], self.std[cls][i]),
                                                 math.e)
        self.class_prob = {cls: math.e ** self.class_prob[cls] for cls in self.class_prob}
        return self.class_prob

    def predict(self, x):
        pred = []
        x = x.to_numpy()
        for x_item in x:
            pred_class = None
            max_prob = 0
            for cls, prob in self.predict_proba(x_item).items():
                if prob > max_prob:
                    max_prob = prob
                    pred_class = cls
            pred.append(pred_class)
        return pred


def dataset_iris():
    iris_data = pd.read_csv("./data/iris.txt", usecols=[0, 1, 2, 3], header=None)
    iris_label = pd.read_csv("./data/iris.txt", usecols=[4], header=None)

    logging.debug(iris_data)
    logging.debug(iris_label)

    logging.info("CLASSIFICATION...")
    train_data, test_data, train_label, test_label = train_test_split(iris_data, iris_label, test_size=0.3,
                                                                      random_state=0)
    nb_gaussian = NaiveBayesGaussian()
    nb_gaussian.fit(train_data, train_label)

    logging.info("PREDICTION...")
    result = nb_gaussian.predict(test_data)
    logging.debug(result)

    iris_classes = {'setosa': 1, 'versicolor': 2, 'virginica': 3}
    result = [iris_classes[item] for item in result]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # multiplica por 30 simplesmente para melhorar a exibição no gráfico
    ax.scatter(test_data[0], test_data[1], test_data[2], s=test_data[3]*30, c=result, marker="o")

    plt.title("Dataset: Iris")
    plt.show()


def dataset_haberman():
    haberman_data = pd.read_csv("./data/haberman.data", usecols=[0, 1, 2], header=None)
    haberman_label = pd.read_csv("./data/haberman.data", usecols=[3], header=None)

    logging.debug(haberman_data)
    logging.debug(haberman_label)

    logging.info("CLASSIFICATION...")
    train_data, test_data, train_label, test_label = train_test_split(haberman_data, haberman_label, test_size=0.3,
                                                                      random_state=0)
    nb_gaussian = NaiveBayesGaussian()
    nb_gaussian.fit(train_data, train_label)

    logging.info("PREDICTION...")
    result = nb_gaussian.predict(test_data)
    logging.debug(result)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(test_data[0], test_data[1], test_data[2], c=result, marker="o")

    plt.title("Dataset: Haberman")
    plt.show()


def dataset_container():
    balance_data = pd.read_csv("./data/balance-scale.data", usecols=[1, 2, 3, 4], header=None)
    balance_label = pd.read_csv("./data/balance-scale.data", usecols=[0], header=None)

    logging.debug(balance_data)
    logging.debug(balance_label)

    logging.info("CLASSIFICATION...")
    train_data, test_data, train_label, test_label = train_test_split(balance_data, balance_label, test_size=0.3,
                                                                      random_state=0)
    nb_gaussian = NaiveBayesGaussian()
    nb_gaussian.fit(train_data, train_label)

    logging.info("PREDICTION...")
    result = nb_gaussian.predict(test_data)
    logging.debug(result)

    balance_classes = {'L': 1, 'B': 2, 'R': 3}
    result = [balance_classes[item] for item in result]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # multiplica por 30 simplesmente para melhorar a exibição no gráfico
    ax.scatter(test_data[1], test_data[2], test_data[3], s=test_data[4] * 30, c=result, marker="o")

    plt.title("Dataset: Balance")
    plt.show()


logging.basicConfig(stream=sys.stderr, level=logging.INFO)

dataset_iris()

dataset_haberman()

dataset_container()
