# Nikodem Malinowski 189018

import math
import time
from copy import copy, deepcopy
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

n = 918
a1 = 3  
a2 = -1
a3 = -1

# n = 9*1*8
# a1 = 3
# a2 = -1
# a3 = -1

def multiply_matrix(A,B,bool):
    if bool:
        C = [[0 for x in range(n)] for y in range(n)]
        for row in range(n):
            for col in range(n):
                for elt in range(n):
                    C[row][col] += A[row][elt] * B[elt][col]
        return C
    else:
        C = [0 for x in range(n)]
        for row in range(n):
                for elt in range(n):
                    C[row] += A[row][elt] * B[elt]
        return C

def sum_matrix(A,B,bool):
    if bool:
        C = [[0 for x in range(n)] for y in range(n)]
        for row in range(n):
            for col in range(n):
                C[row][col] = A[row][col] + B[row][col]
        return C
    else:
        C = [0 for x in range(n)]
        for row in range(n):
                C[row] = A[row] + B[row]
        return C

def sub_matrix(A,B,bool):
    if bool:
        C = [[0 for x in range(n)] for y in range(n)]
        for row in range(n):
            for col in range(n):
                C[row][col] = A[row][col] - B[row][col]
        return C
    else:
        C = [0 for x in range(n)]
        for row in range(n):
                C[row] = A[row] - B[row]
        return C

def norm_frob(wektor):
        suma = 0
        for col in range(n):
            suma = suma + wektor[col]*wektor[col]
        return math.sqrt(suma)


def forward_sub(A,b):
    x = [0 for x in range(n)]

    for i in range(0,n):
        sum = b[i]
        for j in range(0,i):
            sum = sum - A[i][j]*x[j]
        sum = sum/A[i][i]

        x[i] = sum
    return x

def backward_sub(A,b):
    x = [0 for x in range(n)]


    for i in range(n-1,-1,-1):
        sum = b[i]
        for j in range(n-1,i,-1):
            sum = sum - A[i][j]*x[j]
        sum = sum/A[i][i]

        x[i] = sum
    return x

wykres_it = []
wykres_blad = []
wykres_it2 = []
wykres_blad2 = []

A = [[0 for x in range(n)] for y in range(n)]

b = [0 for x in range(n)]

x_new = [1 for x in range(n)]
x_gaus = [1 for x in range(n)]

for i in range(0,n):
    b[i] = math.sin((i+1)*(9+1))

# a1
for i in range (0,n):
    A[i][i] = a1

# a2
for i in range(0,n-1):
    A[i][i+1] = a2
    A[i+1][i] = a2

# a3
for i in range(0,n-2):
    A[i+2][i] = a3
    A[i][i+2] = a3


treshold = 1e-9

U = [[0 for x in range(n)] for y in range(n)]

for i in range (0,n):
    for j in range (1+i,n):
        U[i][j] = -A[i][j]

L = [[0 for x in range(n)] for y in range(n)]

for i in range (1,n):
    for j in range (0,i):
        L[i][j] = -A[i][j]

D = [[0 for x in range(n)] for y in range(n)]

for i in range (0,n):
    D[i][i] = A[i][i]


D_inv = [[0 for x in range(n)] for y in range(n)]

for i in range (0,n):
    D_inv[i][i] = 1/D[i][i]


# Metoda Jacobiego

comp1 = multiply_matrix(D_inv,sum_matrix(L,U,True),True)


comp2 = multiply_matrix(D_inv,b,False)

it_jac = 0

norm = []

blad = 1
start = time.time()
while True:
    it_jac += 1


    x_new = sum_matrix(multiply_matrix(comp1,x_new,False),comp2,False)

    norm = sub_matrix(multiply_matrix(A,x_new,False),b,False)
    blad = norm_frob(norm)

    wykres_it.append(it_jac)
    wykres_blad.append(blad)

    if blad < treshold or it_jac==1000:
        break;

end = time.time()

plt.semilogy(wykres_it,wykres_blad)
plt.title("Jacobi")
plt.xlabel("iteracja")
plt.ylabel("Błąd")
#
plt.show()

print("Jaccobi ilosc iteracji ",it_jac," w czasie: ",end-start)

print("Wektor odp")

for i in range(0,n):
    print(x_new[i],end = " ")

print()
print()


# metoda Gausa

it_gaus = 0
comp1 = sub_matrix(D,L,True)
comp2 = forward_sub(sub_matrix(D,L,True),b)

start = time.time()

while True:
    it_gaus += 1
    x_gaus = sum_matrix(forward_sub(comp1,multiply_matrix(U,x_gaus,False)),comp2,False)

    norm = sub_matrix(multiply_matrix(A,x_gaus,False),b,False)

    blad = norm_frob(norm)

    wykres_it2.append(it_gaus)
    wykres_blad2.append(blad)

    if blad < treshold or it_gaus==1000:
        break;
end = time.time()

print("Gauss ilosc iteracji ",it_gaus," w czasie: ",end-start)

print("Wektor odp")

for i in range(0,n):
    print(x_gaus[i],end = " ")

print()
print()

plt.semilogy(wykres_it2,wykres_blad2)
plt.title("Gauss")
plt.xlabel("iteracja")
plt.ylabel("Błąd")

plt.show()

# metoda faktoryzacji LU

U = deepcopy(A)
L = [[0 for x in range(n)] for y in range(n)]

for i in range (0,n):
    L[i][i] = 1

start = time.time()

for k in range(0,n-1):
    for j in range(k+1,n):
        L[j][k]=U[j][k]/U[k][k]
        for i in range(k,n):
            U[j][i]=U[j][i]-L[j][k]*U[k][i]

y = forward_sub(L,b)
x = backward_sub(U,y)

end = time.time()

norm = sub_matrix(multiply_matrix(A,x,False),b,False)

blad = norm_frob(norm)

print("LU  w czasie: ",end-start)
print("z bledem ",blad)
print("Wektor odp")

for i in range(0,n):
    print(x[i],end = " ")

print()
print()

# # print
# print("Macierz A")
# print()
# for i in range (0,n):
#     for j in range (0,n):
#         print(L[i][j], end = " ")
#     print()

# print("Wektor b")
#
# for i in range(0,n):
#     print(b[i],end = " ")
#
# print()
# print()
#
#
#
#
# print("Wektor X")
#
# for i in range(0,n):
#     print(x[i],end = " ")
# #
# print()
# print()
