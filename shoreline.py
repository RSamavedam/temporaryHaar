import numpy as np

#helper function
def get_harr_matrix(desired_dimension):
    current_dimension = 2
    harr_matrices = {2: np.array([[1, 1], [1, -1]])}

    while current_dimension < desired_dimension:
        top_half_matrix = np.kron(harr_matrices[current_dimension], np.array([1, 1]))
        bottom_half_matrix = np.kron(np.identity(current_dimension), np.array([1, -1]))
        current_dimension *= 2
        harr_matrices[current_dimension] = np.concatenate((top_half_matrix, bottom_half_matrix), axis=0)
        # print(harr_matrices[current_dimension])
        # print(np.linalg.det(harr_matrices[current_dimension]))
        # harr_matrices[current_dimension] /= np.linalg.det(harr_matrices[current_dimension])

    desired_harr_matrix = harr_matrices[current_dimension] #/ np.linalg.det(harr_matrices[current_dimension])
    for i in range(desired_harr_matrix.shape[0]):
        norm = np.linalg.norm(desired_harr_matrix[i])
        for j in range(desired_harr_matrix.shape[1]):
            desired_harr_matrix[i][j] /= norm

    return desired_harr_matrix


def haar_old(input):
    #assume input is a list
    #convert it to numpy array
    input = np.array(input)
    desired_dimension = len(input)
    desired_harr_matrix = get_harr_matrix(desired_dimension)
    wavelet = np.matmul(desired_harr_matrix, input)
    return tuple(wavelet)


def ihaar_old(wavelet):
    #assume input is a list
    #convert it to numpy array
    wavelet = np.array(wavelet)
    desired_dimension = len(wavelet)
    desired_harr_matrix = get_harr_matrix(desired_dimension)
    desired_inverse_harr_matrix = np.transpose(desired_harr_matrix)
    original = np.matmul(desired_inverse_harr_matrix, wavelet)
    return tuple(original)

#everything above this line works (it is based directly on the matrix transform definition from wikipedia) but is more inefficient compared to the table-based method detailed in the handout. This method is implemented below.

#helper function
def get_left_table(input):
    left_table = []
    remaining = input[:]
    left_table.append(remaining[:])
    denominator = np.sqrt(2)
    while (len(remaining) != 1):
        new_remaining = []
        for i in range(0, len(remaining), 2):
            new_remaining.append((remaining[i] + remaining[i+1]) / denominator)
        remaining = new_remaining[:]
        left_table.append(remaining[:])
    left_table.reverse()
    return left_table

def haar(input):
    transform = []
    left_table = get_left_table(input)
    transform.append(left_table[0][0])
    left_table = left_table[1:]
    for table in left_table:
        for i in range(0, len(table), 2):
            transform.append((table[i] - table[i+1]) / np.sqrt(2))
    #print(transform)
    return transform

def ihaar(input):
    left_table = []
    left_table.append([input[0]])
    idx = 1
    while len(left_table[-1]) < len(input):
        new_table = []
        for entry in left_table[-1]:
            new_table.append((entry + input[idx]) / np.sqrt(2))
            new_table.append((entry - input[idx]) / np.sqrt(2))
            idx += 1
        left_table.append(new_table[:])
    return left_table[-1]

def merge(wave1, wave2):
    combined = []
    combined.append((wave1[0] + wave2[0]) / np.sqrt(2))
    combined.append((wave1[0] - wave2[0]) / np.sqrt(2))
    wave1_position = 1
    wave2_position = 1
    run_length = 1
    while (wave1_position < len(wave1)) and (wave2_position < len(wave2)):
        for idx in range(wave1_position, wave1_position + run_length):
            combined.append(wave1[idx])
        wave1_position += run_length
        for idx in range(wave2_position, wave2_position + run_length):
            combined.append(wave2[idx])
        wave2_position += run_length
        run_length *= 2
    return combined

#Run some tests

a = [1, 3, 5, 11, 12, 13, 0, 1] #length of this needs to be a power of 2
first = a[:len(a) // 2]
second = a[len(a) // 2:]

print("Testing haar and inverse haar")
print("-" * len("Testing haar and inverse haar"))
print("Array: ", end="")
print(a)
print("Haar Transform: ", end="")
print(haar(a))
print("Haar Transform (Wikipedia): ", end="")
print(haar_old(a))
b = haar(a)
c = ihaar(b)
print("Inverse Haar Transform: ", end="")
print(c)
b = haar_old(a)
c = ihaar_old(b)
print("Inverse Haar Transform (Wikipedia): ", end="")
print(c)

print()

print("Testing merge")
print("-" * len("Testing haar and inverse haar"))
print("Full Array: ", end="")
print(a)
print("Full Aray Haar Transform: ", end="")
print(haar(a))
first_harr = haar(first)
second_harr = haar(second)
merged_harr = merge(first_harr, second_harr)
print("Result from merging haar transform of two halves: ", end="")
print(merged_harr)

"""
Answers to Written Questions:

NOTE: Coding implementation I'm referring to are the haar and ihaar functions that are more based on the handout

2. I used a python list to construct / store the output of haar as it allows me to add new elements relatively easily / it doesn't
require me to know how much space I need to know beforehand.

3. The input needs to be a power of 2 because of how the Haar Matrix is defined with the Kronecker Product / each array at a given
resoluion level is half as small as the array associated with the next highest resolution level (see handout). If the input is not a
power of 2 I would zero pad it until it is.

5. Realized that the table / tree structure needs to be concatenated in a certain kind of way to get the last N - 2 values.

6. You could possibly create a compression algorithm by taking in a signal input, taking its haar transform,
zeroing out the the k smallest values (by magnitude) and storing that or zeroing out the last k values of the haar wavelet and storing
that. You can run the inverse haar transform on these zeroed out / simplified haar wavelets to reconstitute an approximation
of the original. As an error metric, something like mean-squared-error (MSE) could work for this context where you are
comparing the reconstituted approximation to the original.

7. In general, it's better to train on higher level feature representations than with raw data as it provides a shortcut of sorts
to the learning process. In applications like computer vision, most of the model (e.g. resnet) is used to create a good generic
representation of the input data, and relatively little processing / layers are needed to turn that representation into some prediction
for a task. With a good handcrafted representation as input, the model is more likely to "think at a higher level," likely reducing
how big our models need to be / making it better for us if we have fewer training samples.

"""
