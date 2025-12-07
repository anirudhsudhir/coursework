import numpy as np
from collections import Counter

def get_entropy_of_dataset(data: np.ndarray) -> float:
    """
    Calculate the entropy of the entire dataset using the target variable (last column).
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
    
    Returns:
        float: Entropy value calculated using the formula: 
               Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i
    
    Example:
        data = np.array([[1, 0, 'yes'],
                        [1, 1, 'no'],
                        [0, 0, 'yes']])
        entropy = get_entropy_of_dataset(data)
        # Should return entropy based on target column ['yes', 'no', 'yes']
    """
    target_column = data[:, -1]
    unique_classes, counts = np.unique(target_column, return_counts=True)
    total_samples = len(target_column)
    
    entropy = 0.0
    for count in counts:
        probability = count / total_samples
        if probability > 0:
            entropy -= probability * np.log2(probability)
    
    return entropy


def get_avg_info_of_attribute(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the average information (weighted entropy) of a specific attribute.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
        attribute (int): Index of the attribute column to calculate average information for
    
    Returns:
        float: Average information calculated using the formula:
               Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) 
               where S_v is subset of data with attribute value v
    
    Example:
        data = np.array([[1, 0, 'yes'],
                        [1, 1, 'no'],
                        [0, 0, 'yes']])
        avg_info = get_avg_info_of_attribute(data, 0)  # For attribute at index 0
        # Should return weighted average entropy for attribute splits
    """
    attribute_column = data[:, attribute]
    unique_values = np.unique(attribute_column)
    total_samples = len(data)
    
    avg_info = 0.0
    
    for value in unique_values:
        subset_mask = attribute_column == value
        subset_data = data[subset_mask]
        subset_size = len(subset_data)
        proportion = subset_size / total_samples
        subset_entropy = get_entropy_of_dataset(subset_data)
        avg_info += proportion * subset_entropy
    
    return avg_info


def get_information_gain(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the Information Gain for a specific attribute.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
        attribute (int): Index of the attribute column to calculate information gain for
    
    Returns:
        float: Information gain calculated using the formula:
               Information_Gain = Entropy(S) - Avg_Info(attribute)
               Rounded to 4 decimal places
    
    Example:
        data = np.array([[1, 0, 'yes'],
                        [1, 1, 'no'],
                        [0, 0, 'yes']])
        gain = get_information_gain(data, 0)  # For attribute at index 0
        # Should return the information gain for splitting on attribute 0
    """
    dataset_entropy = get_entropy_of_dataset(data)
    avg_info = get_avg_info_of_attribute(data, attribute)
    information_gain = dataset_entropy - avg_info
    return round(information_gain, 4)


def get_selected_attribute(data: np.ndarray) -> tuple:
    """
    Select the best attribute based on highest information gain.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
    
    Returns:
        tuple: A tuple containing:
            - dict: Dictionary mapping attribute indices to their information gains
            - int: Index of the attribute with the highest information gain
    
    Example:
        data = np.array([[1, 0, 2, 'yes'],
                        [1, 1, 1, 'no'],
                        [0, 0, 2, 'yes']])
        result = get_selected_attribute(data)
        # Should return something like: ({0: 0.123, 1: 0.456, 2: 0.789}, 2)
        # where 2 is the index of the attribute with highest gain
    """
    num_attributes = data.shape[1] - 1
    gain_dict = {}
    
    for attribute_idx in range(num_attributes):
        gain = get_information_gain(data, attribute_idx)
        gain_dict[attribute_idx] = gain
    
    selected_attribute = max(gain_dict.keys(), key=lambda k: gain_dict[k])
    
    return gain_dict, selected_attribute