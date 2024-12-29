import numpy as np

def load_data(path):
    data = np.loadtxt(path)
    labels = data[:, 0].astype(int)
    features = data[:, 1:]
    return labels, features

class KernelPerceptron:
    def __init__(self, kernel):
        self.kernel = kernel
        self.alpha = []
        self.support_vectors = []
        self.support_labels = []
    
    def fit(self, X, y, epochs=3):
        n_samples = len(y)
        self.alpha = np.zeros(n_samples)
        for _ in range(epochs): # repeat whole set
            for t in range(n_samples):
                prediction = np.sign(np.sum(self.alpha*y*self.kernel(X, X[t])))
                if prediction != y[t]:
                    self.alpha[t] = y[t]
                    self.support_vectors.append(X[t]) # item not in this class
                    self.support_labels.append(y[t]) # otherwise as 0

    def predict(self, X):
        predictions = []
        for x in X:
            score = np.sum(
                [
                    alpha * label * self.kernel(support_vector, x)
                    for alpha, support_vector, label in zip(
                        self.alpha, self.support_vectors, self.support_labels
                    )
                ]
            )
            predictions.append(np.sign(score))
        return np.array(predictions)
    
def polynomial_kernel(X, Y, degree=3):
    return np.dot(X, Y.T)**degree

def one_vs_rest(X_train, y_train, X_test, y_test, k, kernel, epochs=3):
    classifiers = []
    for c in range(k):
        y_binary = np.where(y_train == c, 1, -1) # change the original tag into binary
        classifier = KernelPerceptron(kernel)
        classifier.fit(X_train, y_binary, epochs=epochs)
        classifiers.append(classifier)

    confidence_scores = np.array([clf.predict(X_test) for clf in classifiers])
    predictions = np.argmax(confidence_scores, axis=0)
    return predictions
    