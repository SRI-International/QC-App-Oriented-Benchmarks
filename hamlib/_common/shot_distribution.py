'''
Hamiltonian Simulation Benchmark Program - Qiskit
(C) Quantum Economic Development Consortium (QED-C) 2025.
'''

# ### TODO
# 
# - Remove 0's from incoming arrays of groups and circuits
# 
# - Determine how to set max_buckets; num_groups / 5 e.g. or some other function
# 
# - construct arrays of circuits from the oringinal indices; number of arrays should be same as len(bucket_averages)
# 
# - execute arrays of circuits with bucket averages  (using existing code in hamlib_simulation_benchmark.py
# 
# - 

import numpy as np
import bisect
import random

import matplotlib.pyplot as plt

def auto_select_tolerance(num_list):
    """
    Automatically selects a reasonable tolerance based on the natural gaps in sorted data.
    Uses interquartile range (IQR) to determine a reasonable threshold.
    
    Args:
        num_list (list): List of integers to be grouped.

    Returns:
        float: Auto-detected tolerance value.
    """
    num_list = sorted(num_list, reverse=True)
    
    # Compute the differences between consecutive values
    gaps = np.diff(num_list)

    # Use Interquartile Range (IQR) to find a typical gap size
    q1, q3 = np.percentile(gaps, [25, 75])
    iqr = q3 - q1
    suggested_tolerance = q3 + 1.5 * iqr  # Tukey's rule for outliers

    return max(suggested_tolerance, np.std(num_list) * 0.5)  # Ensure a reasonable lower bound

def bucket_numbers_auto_fast(num_list, max_buckets):
    """
    Optimized Auto-Tolerance bucketing with indices that maintain the exact nested structure.

    Args:
        num_list (list): List of integers to be grouped.
        max_buckets (int): Maximum number of buckets allowed.

    Returns:
        tuple: (buckets, bucket_indices)
            - buckets: Nested list of grouped shot values.
            - bucket_indices: Nested list of original indices matching the structure of buckets.
    """
    if not num_list:
        return [], []

    # Store numbers with their original indices before sorting
    indexed_nums = sorted(enumerate(num_list), key=lambda x: x[1], reverse=True)
    sorted_nums = [num for _, num in indexed_nums]
    original_indices = [idx for idx, _ in indexed_nums]

    # Automatically determine a tolerance
    tolerance = auto_select_tolerance(sorted_nums)

    # Initialize buckets and corresponding index storage
    buckets = [[sorted_nums[0]]]
    bucket_indices = [[original_indices[0]]]  # Ensure structure is nested

    for i in range(1, len(sorted_nums)):
        num = sorted_nums[i]
        idx = original_indices[i]
        placed = False

        # Use binary search to find a suitable bucket
        for j in range(len(buckets)):
            if abs(num - np.mean(buckets[j])) <= tolerance:
                bisect.insort(buckets[j], num)  # Insert number in sorted order
                bisect.insort(bucket_indices[j], idx)  # Insert index in the same order
                placed = True
                break
        
        # Create a new bucket if needed
        if not placed:
            if len(buckets) < max_buckets:
                buckets.append([num])
                bucket_indices.append([idx])
            else:
                # Add to the closest existing bucket
                closest_bucket = min(range(len(buckets)), key=lambda j: abs(num - np.mean(buckets[j])))
                bisect.insort(buckets[closest_bucket], num)
                bisect.insort(bucket_indices[closest_bucket], idx)

    return buckets, bucket_indices

####################################################################

def kmeans_clustering(num_list, max_buckets, max_iters=100):
    """
    Simple K-Means clustering algorithm.

    Args:
        num_list (list): List of integers to be clustered.
        max_buckets (int): Maximum number of clusters.
        max_iters (int): Maximum iterations for convergence.

    Returns:
        list: A list of lists representing the clusters.
    """
    if not num_list or max_buckets < 1:
        return []

    num_array = np.array(num_list)

    # Randomly initialize cluster centers
    centroids = random.sample(list(num_array), min(max_buckets, len(num_list)))

    for _ in range(max_iters):
        # Assign each number to the closest centroid
        clusters = {i: [] for i in range(len(centroids))}
        for num in num_array:
            closest = min(range(len(centroids)), key=lambda i: abs(num - centroids[i]))
            clusters[closest].append(num)

        # Update centroids (mean of each cluster)
        new_centroids = [np.mean(clusters[i]) if clusters[i] else centroids[i] for i in range(len(centroids))]

        # If centroids don't change, we are done
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    # Convert dictionary to list of lists
    return list(clusters.values())


def bucket_numbers_kmeans(num_list, max_buckets):
    """
    K-Means clustering that tracks original indices.

    Args:
        num_list (list): List of integers to be grouped.
        max_buckets (int): Maximum number of clusters.

    Returns:
        tuple: (buckets, bucket_indices)
            - buckets: Nested list of grouped shot values.
            - bucket_indices: Nested list of original indices.
    """
    if not num_list or max_buckets < 1:
        return [], []

    # Store numbers with their original indices
    indexed_nums = [(idx, num) for idx, num in enumerate(num_list)]
    
    # Extract only the values for clustering
    num_values = [num for _, num in indexed_nums if num != 0]

    # Run K-Means Clustering
    clustered_values = kmeans_clustering(num_values, max_buckets)

    # Match the values back to their original indices
    buckets = []
    bucket_indices = []
    
    for cluster in clustered_values:
        bucket = []
        indices = []

        for num in cluster:
            # Find and remove the first occurrence of this value in indexed_nums
            for idx, value in indexed_nums:
                if value == num:
                    bucket.append(value)
                    indices.append(idx)
                    indexed_nums.remove((idx, value))  # Avoid duplicate matching
                    break
        if bucket:
            buckets.append(bucket)
            bucket_indices.append(indices)

    return buckets, bucket_indices


####################################################################

def compute_bucket_averages(buckets):
    """
    Computes the average number of shots for each bucket.
    
    Args:
        buckets (list of lists): The grouped buckets of shot numbers.
    
    Returns:
        list: An array where each entry represents the average number of shots for that bucket.
    """
    return [sum(bucket) // len(bucket) for bucket in buckets]


def visualize_buckets(num_list, buckets, title):
    """
    Plots the grouped numbers as clusters in a scatter plot.

    Args:
        num_list (list): Original list of numbers.
        buckets (list): List of grouped buckets.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(8, 5))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'v']

    for i, bucket in enumerate(buckets):
        bucket_color = colors[i % len(colors)]
        bucket_marker = markers[i % len(markers)]
        plt.scatter(bucket, [i] * len(bucket), color=bucket_color, label=f'Bucket {i+1}', marker=bucket_marker, s=100)

    plt.xlabel("Value")
    plt.ylabel("Bucket Index")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


####################################################################
# ### Test Shot Distribution Functions

# if main, execute method
if __name__ == '__main__':

    num_shots_list = [6962, 1159, 216, 111, 1159, 57, 57, 111, 57, 111]
    num_shots_list = [751, 274, 113, 160, 113, 3101, 354, 46, 0, 0, 113, 400, 1734, 113, 0, 274, 46, 2134, 274]
    print("")

    # Use auto-selected tolerance method
    buckets_auto, indices_auto = bucket_numbers_auto_fast(num_shots_list, max_buckets=3)
    print("Buckets (Auto-Tolerance):", buckets_auto)
    print("Buckets Averaged (Auto-Tolerance): ", compute_bucket_averages(buckets_auto))
    print("Original Indices (Auto-Tolerance): ", indices_auto)
    print("")
    visualize_buckets(num_shots_list, buckets_auto, title="Buckets Using Auto-Selected Tolerance")
    print("")

    # Use K-Means clustering
    buckets_kmeans, indices_kmeans = bucket_numbers_kmeans(num_shots_list, max_buckets=3)
    print("Buckets (K-Means Clustering):", buckets_kmeans)
    print("Buckets Averaged (K-Means):", compute_bucket_averages(buckets_kmeans))
    print("Original Indices (K-Means):", indices_kmeans)
    print("")
    visualize_buckets(num_shots_list, buckets_kmeans, title="Buckets Using K-Means Clustering")
    print("")


