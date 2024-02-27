import copy


def alg_dop(A: list, i: int, j: int):
    M = len(A)
    Mi = [item for item in range(M) if item != i]  # индексы строк
    Mj = [item for item in range(M) if item != j]  # индексы столбцов
    new_matr_ = new_matr(A, Mi, Mj)
    minor_ = determinant(new_matr_)
    alg_dop_ = (-1) ** (i + j) * minor_
    return alg_dop_


def inversion_matrix(A: list):
    """Поиск обратной матрицы"""
    matr_alg_dop_ = [[0 for _ in range(len(A))] for _ in range(len(A))]
    det_A = determinant(A)
    if det_A == 0:
        return []
    for i in range(len(A)):
        for j in range(len(A)):
            matr_alg_dop_[i][j] = alg_dop(A, i, j) / det_A
    return matr_alg_dop_


def transport_matrix(matr: list) -> list:
    """Транспонирование матрицы"""
    transp_m = [[0 for _ in range(len(matr))] for _ in range(len(matr[0]))]
    for i in range(len(matr)):  # проход по строкам
        for j in range(len(matr[0])):  # проход по столбцам
            transp_m[j][i] = matr[i][j]
    return transp_m


def find_nonull_(A: list, k: int):
    new_A = transport_matrix(A)
    A_k = new_A[k]
    id_ = [i for i in range(len(A_k)) if A_k[i] != 0]
    if not id_:
        return -1, False
    return id_[0], True


def determinant(a):
    """Поиск определителя матрицы"""
    matr = copy.deepcopy(a)
    n = len(matr)
    per = 0  # количество пересатновок в матрице
    k = 0
    while k <= n - 1:
        if matr[k][k] == 0:  # перемещение строки, если у неё нулевые элементы на диагонали
            id_, flag = find_nonull_(matr, k)
            if not flag:
                return 0
            new_row = matr[id_]
            matr[id_] = matr[k]
            matr[k] = new_row
            per += 1
            continue
        k += 1

    for k in range(0, n):

        for i in range(k + 1, n):
            if matr[i][k] != 0.0:
                # Если не нулевой диагональный элемент, то вычисляем лямбду
                lam = matr[i][k] / matr[k][k]
                # вычисляем новую строку матрицы
                for j in range(k + 1, n):
                    matr[i][j] = matr[i][j] - lam * matr[k][j]
                continue

    det = 1
    for k in range(0, n):
        det *= matr[k][k]
    det *= (-1) ** per
    return det


def mult_lists(a: list, b: list, k: int, n: int):
    """Скалярное произведение вектора матрицы на другой вектор"""
    sum_ = 0
    for j in range(k + 1, n):
        sum_ += a[k][j] * b[j]
    return sum_


def gaussMeth(matr: list, b_: list):
    """Метод Гаусса для решения СЛАУ"""

    a = copy.deepcopy(matr)
    b = copy.deepcopy(b_)
    n = len(b)

    k = 0
    while k < n - 1:
        if a[k][k] == 0:  # перемещение строки, если у неё нулевые элементы на диагонали
            id_, flag = find_nonull_(a, k)
            if not flag:
                return 0
            new_row = a[id_]
            a.pop(id_)
            a.append(a[k])
            a.pop(k)
            a.insert(0, new_row)
            k -= 1
            continue
        k += 1

    # Прямой ход метода Гаусса с выбором ведущего элемента
    for k in range(0, n - 1):
        for i in range(k + 1, n):
            if a[i][k] != 0.0:
                # Если не нулевой диагональный элемент, то вычисляем лямбду
                lam = a[i][k] / a[k][k]
                # вычисляем новую строку матрицы
                for j in range(k + 1, n):
                    a[i][j] = a[i][j] - lam * a[k][j]
                # вычисляем новвый вектор b
                b[i] = b[i] - lam * b[k]

    # обратный ход метода Гаусса
    for k in range(n - 1, -1, -1):  # от n-1 до -1 с шагом -1
        b[k] = (b[k] - mult_lists(a, b, k, n)) / a[k][k]

    return b  # решение системы


def matrix_by_id_col(matr: list, list_comb: list):
    n = len(matr)
    new_matr_ = [[0 for _ in range(len(list_comb))] for _ in range(n)]
    for i in range(n):
        k = 0
        for j in list_comb:
            new_matr_[i][k] = matr[i][j]
            k += 1
    return new_matr_


def new_matr(matr: list, list_col: list, list_row: list):
    """Матрица по спискам номеров столбцов и строк"""
    new_matr_ = [[0 for _ in range(len(list_col))] for _ in range(len(list_row))]
    i = 0
    for r in list_row:
        j = 0
        for c in list_col:
            new_matr_[i][j] = matr[r][c]
            j += 1
        i += 1
    return new_matr_


def new_vec(vec: list, list_el: list):
    new_vec_ = [0 for _ in range(len(list_el))]
    i = 0
    for k in list_el:
        new_vec_[i] = vec[k]
        i += 1
    return new_vec_


def mult_vec(a: list, b: list):
    """Скалярное произведение векторов"""
    sum_ = 0
    for i in range(len(a)):
        sum_ += a[i] * b[i]
    return sum_


def mult_matr(A: list, B: list):
    """Произведение матриц"""
    mult_matr_ = [[0 for _ in range(len(A[0]))] for _ in range(len(B))]
    new_A_transp = transport_matrix(A)
    for i in range(len(B)):
        for j in range(len(new_A_transp)):
            mult_matr_[i][j] = mult_vec(B[i], new_A_transp[j])
    return mult_matr_


def mult_vec_matr(c: list, BA: list):
    """Произведение вектора на матрицу"""
    transp_AB = transport_matrix(BA)
    mult_ = [0 for _ in range(len(transp_AB))]

    for i in range(len(transp_AB)):
        mult_[i] = mult_vec(c, transp_AB[i])

    return mult_


def diff_vec(cL: list, cN: list):
    vec_diff = [0 for _ in range(len(cL))]
    for i in range(len(cL)):
        vec_diff[i] = cL[i] - cN[i]
    return vec_diff


def mult_scal_vec(scale: float, u: list):
    new_vec_ = [0 for _ in range(len(u))]
    for i in range(len(u)):
        new_vec_[i] = scale * u[i]
    return new_vec_


A = [[1,2,0], [4,5,6], [7,8,9]]
b = [1, 2, 3]
print(determinant(A))
print(inversion_matrix(A))
