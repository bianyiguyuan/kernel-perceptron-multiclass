import numpy as np
from random import shuffle

def load_data(path):
    data = np.loadtxt(path)
    labels = data[:, 0].astype(int)
    features = data[:, 1:]
    return labels, features

def train_test_split(X, y, test_size=0.2, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    combined = list(zip(X, y))
    shuffle(combined)
    X, y = zip(*combined)

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = np.array(X[:split_idx]), np.array(X[split_idx:])
    y_train, y_test = np.array(y[:split_idx]), np.array(y[split_idx:])
    return X_train, X_test, y_train, y_test

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

def q3_experiment(path="data/zipcombo.dat.txt", 
                  degrees=[1, 2, 3, 4, 5, 6, 7], 
                  runs=20, 
                  epochs=3):
    
    y, X = load_data(path)
    n_degrees = len(degrees)

    train_errors = np.zeros((runs, n_degrees))
    test_errors = np.zeros((runs, n_degrees))

    for run in range(runs):
        print(f"Run {run + 1}/{runs}...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=run)
        
        for idx, degree in enumerate(degrees):
            print(f"Training with polynomial degree {degree} ({idx + 1}/{n_degrees})...")
            kernel = lambda X, Y: polynomial_kernel(X, Y, degree)
            perceptron = KernelPerceptron(kernel)
            
            # Train the perceptron
            perceptron.fit(X_train, y_train, epochs=epochs)
            print(f"Finished training for degree {degree}.")
            
            # Predict and compute errors
            train_pred = perceptron.predict(X_train)
            test_pred = perceptron.predict(X_test)
            
            train_errors[run, idx] = np.mean(train_pred != y_train)
            test_errors[run, idx] = np.mean(test_pred != y_test)

            print(f"Train Error={train_errors[run, idx]:.3f}, Test Error={test_errors[run, idx]:.3f}.")
            print(f"Finished testing for degree {degree}.")
    
    train_mean = np.mean(train_errors, axis=0)
    train_std = np.std(train_errors, axis=0)
    test_mean = np.mean(test_errors, axis=0)
    test_std = np.std(test_errors, axis=0)

    print("Train Error (Mean ± Std):")
    for d, mean, std in zip(degrees, train_mean, train_std):
        print(f"d={d}: {mean:.10f} ± {std:.10f}")
    
    print("\nTest Error (Mean ± Std):")
    for d, mean, std in zip(degrees, test_mean, test_std):
        print(f"d={d}: {mean:.3f} ± {std:.3f}")
    
    return 

if __name__ == "__main__":
    q3_experiment()

