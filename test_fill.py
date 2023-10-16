import numpy as np

def fill_sequence(seq):
    arr = np.array(seq)
    non_zero_indices = np.where(arr != 0)[0]

    for i in range(len(non_zero_indices) - 1):
        left_idx = non_zero_indices[i]
        right_idx = non_zero_indices[i + 1]
        midpoint = (left_idx + right_idx) // 2
        
        arr[left_idx:midpoint] = arr[left_idx]
        arr[midpoint:right_idx] = arr[right_idx]

    # For the parts of the array before the first non-zero and after the last non-zero
    if non_zero_indices[0] > 0:
        arr[:non_zero_indices[0]] = arr[non_zero_indices[0]]
    if non_zero_indices[-1] < len(arr) - 1:
        arr[non_zero_indices[-1]+1:] = arr[non_zero_indices[-1]]

    return arr.tolist()

sequence = [0, 0, 0, 1, 0, 0, 0, 0, 2]
print(fill_sequence(sequence))