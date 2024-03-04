from matplotlib import pyplot as plt
import numpy as np



#task 1 :
# A = np.random.rand(200,10)


# mu = np.zeros(A.shape[1])
# for i in range(A.shape[0]):
#     mu += A[i]
#     print(mu)
# mu /= A.shape[0]

# B = np.zeros_like(A)
# for i in range(A.shape[0]):
#     B[i] = A[i] - mu


# c = A - np.mean(A, axis=0)


# are_equal = np.array_equal(B, c)
# print(are_equal)

#----------------------------


#task2 :
I = plt.imread('/Users/hossein/Desktop/CV/0-1/cv-lab1/masoleh_gray.jpg')
I_inverted = I[::-1, :]
#plt.imshow(I_inverted)

concatenated_image = np.concatenate((I, I_inverted), axis=0)
plt.imshow(concatenated_image, cmap='gray')
plt.show()
print(I)
