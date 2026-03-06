

R= Republicday
for n in R:
    print(n)
    A1={n for n in R}
    print("A1")

    print(" ")
    A2=[n for n in R]
    print("A2")

    print(" ")
    print(tuple(A1))
    print(" ")
    print(tuple(A2))