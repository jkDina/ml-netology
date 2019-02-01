def print_(matrix):
    for row in matrix:
        for num in row:
            print(str(num).rjust(5), end=" ")
        print('\n')
		
def strikeout(matrix, a, b):
    n = len(matrix)
    if n == 0:
        raise Exception('matrix is not valid')
    new_matrix = []
    i = 0
    for row in matrix:
        i += 1
        if i == a:
            continue
        cols = []
        j = 0
        for num in row:
            j += 1
            if j == b:
                continue
            cols.append(num)
        new_matrix.append(cols)
    return new_matrix
    
def det(matrix):
    n = len(matrix)
    if n == 0:
        raise Exception('matrix is not valid')
    for el in matrix:
        if len(el) != n:
            raise Exception('matrix is not valid')
    if n == 1:
        return matrix[0][0]
    s = 0
    for pair in enumerate(matrix[0]):
        i = pair[0] + 1
        number = pair[1]
        s += (-1)**(i + 1) * det(strikeout(matrix, 1, i)) * number
    return s

if __name__ == '__main__':
    A1 = [[1, 2], [3, 4]]
    A2 = [[1, 2], [2, 4]]
    A3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    A4 = [[2, 4, 5], [2, 5, 6], [-2, 1, 0]]
    A5 = [[2, -4, 7, -10], [3, 5, 6, -1], [-8, 12, 11, 9], [4, 5, 8, -2]]
    print('A1 = ')
    print_(A1)
    print('A2 = ')
    print_(A2)
    print('A3 = ')
    print_(A3)
    print('A4 = ')
    print_(A4)
    print('A5 = ')
    print_(A5)
    print('det(A1) = ', det(A1))
    print('det(A2) = ', det(A2))
    print('det(A3) = ', det(A3))
    print('det(A4) = ', det(A4))
    print('det(A5) = ', det(A5))
    

          
    
    
    
    
