import numpy as np

def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))

if __name__ == "__main__":
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    print(f"Euclidean distance: {euclidean_distance(v1, v2)}")