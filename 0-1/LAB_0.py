

#Task 1:

def mul(A, B):

    if len(A[0]) != len(B):
        return []


    # for i in range(len(A)):
    #     for j in range(len(B[0])):
    #         for k in range(len(B)):
    #             result[i][j] += A[i][k] * B[k][j]


#found this in internet
    result = [
        [
            sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))
        ]
        for i in range(len(A))
    ]

    return result