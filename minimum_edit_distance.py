def calculate_min_edit_distance(x, y):
    result = []

    n = len(x)
    m = len(y)

    # Инициализация
    for i in range(0, n + 1):
        row = [0] * (m + 1)
        result.append(row)

    for i in range(0, n + 1):
        result[i][0] = i
    for j in range(0, m + 1):
        result[0][j] = j

    # Расчет всех вариантов
    for i in range(0, n + 1):
        for j in range(1, m + 1):
            if x[i - 1] == y[j - 1]:
                result[i][j] = result[i - 1][j - 1]
            else:
                result[i][j] = min(result[i - 1][j] + 1, result[i][j - 1] + 1, result[i - 1][j - 1] + 2)

    print("Minimum edit distance:")
    print(result[n][m])

    i = n
    j = m

    # Вывод на экран
    alignment = []

    while i != 0 or j != 0:

        temp = []

        if i >= 0 and j >= 0 and result[i][j] == result[i - 1][j - 1] and x[i - 1] == y[j - 1]:
            temp.append(x[i - 1])
            temp.append(y[j - 1])
            temp.append("Equal")
            i = i - 1
            j = j - 1
        elif result[i][j] == result[i - 1][j - 1] + 2:
            temp.append(x[i - 1])
            temp.append(y[j - 1])
            temp.append("Substitution")
            i = i - 1
            j = j - 1
        elif result[i][j] == result[i - 1][j] + 1 and i >= 0:
            temp.append(x[i - 1])
            temp.append("*")
            temp.append("Deletion")
            i = i - 1
        elif result[i][j] == result[i][j - 1] + 1 and j >= 0:
            temp.append("*")
            temp.append(y[j - 1])
            temp.append("Insertion")
            j = j - 1

        alignment.append(temp)

    for one in reversed(alignment):
        print(one[0] + " " + one[1] + " " + one[2])


x = "drive"
y = "brief"
calculate_min_edit_distance(x, y)
