import numpy as np
from sklearn.model_selection import StratifiedKFold


class BayesClassifier:
	def __init__(self, priors=None):
		self.priors = None
		
	def fit(self, x, y):
		self.cls = np.unique(y)
		self.n_cls = len(self.cls)
		if self.priors is None:
			self.priors = np.array([np.mean(y == c) for c in self.cls])
			
		self.n_dims = x.shape[1]
		self.mu = np.zeros((self.n_cls, self.n_dims))
		self.sigma = np.zeros((self.n_cls, self.n_dims, self.n_dims))
		for c in range(self.n_cls):
			self.mu[c, ] = np.mean(x[y == c, :], axis=0)
			self.sigma[c, ] = np.cov(x[y == c, :].T)
		
	def predict(self, x_test):
		log_pred = np.zeros((x_test.shape[0], self.n_cls))
		for c in range(self.n_cls):
			log_pred[:, c] = \
				-0.5 * np.sum(np.matmul((x_test - self.mu[c,]), np.linalg.inv(self.sigma[c, ])) *
																		(x_test - self.mu[c, ]), axis=1) - \
							0.5 * np.log(np.linalg.det(self.sigma[c, ])) + np.log(self.priors[c])
		
		return np.argmax(log_pred, axis=1)


def p_entropy(y):
	label = np.unique(y)
	prob = np.array([np.sum(y == l)/float(len(y)) for l in label])
	ent = -np.sum(prob * np.log2(prob))
	return ent


class tree:
	def __init__(self, X, y, prop=None):
		self.X = np.array(X)
		self.y = np.array(y)

		self.feature_dict = {}
		self.labels, self.y = np.unique(y, return_inverse=True)
		self.DT = list()
		if prop is None:
			self.property = np.zeros((self.X.shape[1]))
		else:
			self.property = prop

		for i in range(self.X.shape[1]):
			self.feature_dict.setdefault(i)
			self.feature_dict[i] = np.unique(self.X[:, i])

	def entropy(self, X, y, k, k_v):
		if self.property[k] == 0:
			c1 = (X[X[:, k] == k_v]).shape[0]
			c2 = (X[X[:, k] != k_v]).shape[0]
			D = y.shape[0]
			return c1 * p_entropy(y[X[:, k] == k_v]) / D \
				+ c2 * p_entropy(y[X[:, k] != k_v]) / D
		else:
			c1 = (X[X[:, k] >= k_v]).shape[0]
			c2 = (X[X[:, k] < k_v]).shape[0]
			D = y.shape[0]
			return c1 * p_entropy(y[X[:, k] >= k_v]) / D \
				+ c2 * p_entropy(y[X[:, k] < k_v]) / D

	def makeTree(self,X,y):
		if np.unique(y).size <= 1:
			return y[0]
		
		minp = 10000.0
		m_i, m_j = 0, 0
		for i in range(self.X.shape[1]):
			for j in self.feature_dict[i]:
				p = self.entropy(X, y, i, j)
				if p < minp:
					minp = p
					m_i, m_j = i, j

		if minp == 1:
			return y[0]

		left = []
		right = []
		if self.property[m_i] == 0:
			left = self.makeTree(X[X[:, m_i] == m_j], y[X[:, m_i] == m_j])
			right = self.makeTree(X[X[:, m_i] != m_j], y[X[:, m_i] != m_j])
		else:
			left = self.makeTree(X[X[:, m_i] >= m_j], y[X[:, m_i] >= m_j])
			right = self.makeTree(X[X[:, m_i] < m_j], y[X[:, m_i] < m_j])
		return (m_i, m_j), left, right

	def train(self):
		self.DT = self.makeTree(self.X, self.y)
		
	def predict(self,X):		  
		result = np.zeros(X.shape[0])
		for i in range(X.shape[0]):
			tp = self.DT
			while type(tp) is tuple:
				a, b = tp[0]

				if self.property[a] == 0:
					if X[i][a] == b:
						tp = tp[1]
					else:
						tp = tp[2]
				else:
					if X[i][a] >= b:
						tp = tp[1]
					else:
						tp = tp[2]
			result[i] = self.labels[tp]
		return result


x = []
y = []
for line in open('iris.csv'):
	line = line.strip().split(',')
	if len(line) == 5:
		x.append([float(item) for item in line[:-1]])
		y.append(line[-1])

x = np.array(x)
c = np.unique(y)
c = dict([(_c, i) for i, _c in enumerate(c)])
y = [c[_y] for _y in y]
y = np.array(y)

acc = 0.
for train_index, test_index in StratifiedKFold(n_splits=10).split(x, y):
	x_train, x_test = x[train_index, :], x[test_index, :]
	y_train, y_test = y[train_index], y[test_index]
	model = tree(x_train, y_train)
	model.train()
	z = model.predict(x_test)
	acc += np.mean(z == y_test)
acc /= 10
print('10 fold average accuracy (decision tree) = %.4f' % acc)

acc = 0.
for train_index, test_index in StratifiedKFold(n_splits=10).split(x, y):
	x_train, x_test = x[train_index, :], x[test_index, :]
	y_train, y_test = y[train_index], y[test_index]
	model = BayesClassifier()
	model.fit(x_train, y_train)
	z = model.predict(x_test)
	acc += np.mean(z == y_test)
acc /= 10
print('10 fold average accuracy (naive bayes) = %.4f' % acc)
