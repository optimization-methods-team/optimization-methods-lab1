import copy

import pandas as pd
from tabulate import tabulate
from matrix import *


def sum_(c: list, A: list, i: int, basis: list):
    s = 0
    k = 0
    for j in basis:
        s += c[j] * A[k][i]
        k += 1
    return s


def delta(c: list, A: list, N: list, L: list, Nk: list):
    """
    Вычисление дельта-оценки
    :param c: коэффициенты функции цели
    :param A: матрица коэффициентов
    :param N: список всех индексов
    :param L: список индексов небазисных векторов
    :param Nk: список индексов базисных векторов
    :param M: список индексов строк
    :return: вектор дельта-оценки
    """
    delta_list = [0 for _ in range(len(N))]

    for i in N:
        del_ = c[i] - sum_(c, A, i, Nk)
        delta_list[i] = del_

    return delta_list


def validation_a_delta(delta_: list, A: list, Lk: list):
    for j in Lk:
        if delta_[j] < 0:
            col_otr = [i for i in range(len(A)) if A[i][j] <= 0]
            if len(col_otr) == len(A):  # все коэффициенты столбца отрицательны
                return False

    return True


def permission_column(A: list, c: list, basis: list):
    """
    :param A: матрица коэффициентов
    :return: индекс разрешающего столбца
    """
    N = [i for i in range(len(A[0]))]
    Lk = [item for item in range(len(N)) if item not in basis]
    delta_ = delta(c, A, N, Lk, basis)

    if not validation_a_delta(delta_, A, Lk):
        return [], -2, False  # нужно сообщить о неограниченности задачи

    # используем правило Блэнда (вывод наименьшего коэффициента, для которого дельта-оценка отрицательна
    jk = len(N)
    for j in Lk:
        if delta_[j] < 0:
            if j < jk:
                jk = j

    if jk != len(N):
        return delta_, jk, False

    return delta_, -1, True


def permission_row(A: list, xk: list, j: int, Nk: list):
    """Поиск ведущей строки"""
    list_rel = [0 for _ in range(len(xk))]
    Nu = []
    i = 0
    id_table = []  # для сохранения номера возможных ведущих строк в матрице, тк Nk-индексы базиса,
    # то их индексы могут и не совпасть
    for k in Nk:
        if A[i][j] > 0:
            rel = xk[k] / A[i][j]
            list_rel[k] = rel
            Nu.append(k)
            id_table.append(i)
        i += 1

    min_ = 0
    id_ = 0
    for i in Nu:
        min_ = list_rel[i]
        id_ = i
        break

    id_t = 0
    j = 0
    for i in Nu:
        if list_rel[i] <= min_:
            min_ = list_rel[i]
            if list_rel[id_] == min_:
                if i < id_:
                    id_ = i
            else:
                id_ = i
            id_t = id_table[j]
        j += 1

    return id_, id_t, Nu, list_rel


def print_matrix(A: list, c: list, basis: list, delta_: list, list_rel: list, Nu: list, xk: list, id_t: int, ik: int, jk: int):
    columns = ['x[' + str(i + 1) + ']' for i in range(len(A[0]))]
    columns.append('Theta')
    columns.insert(0, 'X_k')
    columns.insert(0, 'Коэфф. базиса')
    columns.insert(0, 'Базис')

    transp_A = transport_matrix(A)  # к транспонированной матрице проще присоединять векторы

    # отфильтровывываем только нужные theta
    theta_ = [0 for _ in range(len(list_rel))]

    for i in Nu:
        theta_[i] = list_rel[i]

    Nk_Nu = [i for i in basis if i not in Nu]
    for i in Nk_Nu:
        theta_[i] = '-'

    theta_list = []
    xk_list = []
    k_bas = []
    bas_list = []
    for i in basis:
        theta_list.append(theta_[i])
        xk_list.append(xk[i])
        k_bas.append(c[i])
        bas_list.append('x[' + str(i + 1) + ']')

    transp_A.append(theta_list)
    transp_A.insert(0, xk_list)
    transp_A.insert(0, k_bas)
    transp_A.insert(0, bas_list)

    table = transport_matrix(transp_A)
    table.insert(0, columns)

    f_c = ['Ф.ц.']
    f_xk = mult_vec(xk, c)
    f_c.append(f_xk)
    f_c.append('--')

    for i in range(len(c)):
        f_c.append(c[i])

    delta_list = ['delta', '--', '--']
    delta_list += delta_
    table.append(f_c)
    table.append(delta_list)

    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
    print("Пояснения:")
    print("Шаг 1: Вычисляем по формуле из лекции дельта оценку. Выбираем из них отрицательную оценку "
          "с наименьшим коэффициентом и индекс этой оценки будет индексом ведущего столбца")
    print("Если таких отрицательных дельта нет, то можем судить о найденном оптимальном решении")
    print("Нашли индекс для ведущего столбца: ", jk + 1)
    print("Шаг 2: Вычисляем theta по элементам ведущего столбца")
    print("Индекс ведущей строки: ", id_t + 1)
    print("Шаг 3: Поиск ведущего элемента и пересчёт симплекс-таблицы")
    print("Ведущий элемент: ", A[id_t][jk])
    print("Выводим из базиса компоненту: x[", ik + 1, "]")
    print("Вводим в базис компоненту: x[", jk + 1, "]")
    print()


def table_recalculation(A: list, xk: list, basis: list, c: list):
    new_matr_ = [[0 for _ in range(len(A[0]))] for _ in range(len(A))]
    delta_, jk, flag = permission_column(A, c, basis)
    ik, id_t, Nu, list_rel = permission_row(A, xk, jk, basis)

    print_matrix(A, c, basis, delta_, list_rel, Nu, xk, id_t, ik, jk)

    if jk == -2 and flag == False:
        return [], [], [], False

    if flag:
        return A, xk, basis, True

    xkk = [0 for _ in range(len(xk))]
    xkk[jk] = xk[ik] / A[id_t][jk]
    ii = 0
    for i in basis:
        if i == ik:
            ii += 1
            continue
        xkk[i] = xk[i] - xk[ik] * A[ii][jk] / A[id_t][jk]
        ii += 1

    for i in range(len(A)):
        for j in range(len(A[0])):
            if j == jk:
                continue

            if i == id_t:
                if j == jk:
                    continue
                new_matr_[i][j] = A[i][j] / A[id_t][jk]
                continue

            new_matr_[i][j] = A[i][j] - A[id_t][j] * A[i][jk] / A[id_t][jk]

    for i in range(len(A)):
        if i == id_t:
            new_matr_[i][jk] = 1
            continue

    new_basis = [jk if i == ik else i for i in basis]
    return new_matr_, xkk, new_basis, False


def simplex(A: list, c: list, xk: list, basis: list):
    matr_ = copy.deepcopy(A)
    f_c = copy.deepcopy(c)
    xk_ = copy.deepcopy(xk)
    basis_ = copy.deepcopy(basis)
    while 1:
        matr_, xk_, basis_, flag = table_recalculation(matr_, xk_, basis_, f_c)
        if matr_ == [] and xk_ == [] and basis_ == [] and flag == False:
            print("Функция цели неограничена")
            return [], []
        if flag:
            break

    return xk_, basis_


def find_start_bas(A: list, b: list, c: list):
    """Метод искусственного базиса"""
    print("---ПОИСК НАЧАЛЬНОГО ОПОРНОГО ВЕКТОРА---")
    M = [i for i in range(len(A))]
    new_matr_ = copy.deepcopy(A)
    # дополняем матрицу единичной
    for i in M:
        list_ = [1 if i == j else 0 for j in M]
        new_matr_[i] += list_

    xy = [0 for _ in range(len(new_matr_[0]))]  # начальный о.в.

    j = 0
    func_c = [0 for _ in range(len(new_matr_[0]))]
    for i in range(len(A[0]), len(new_matr_[0])):
        xy[i] = b[j]
        func_c[i] = 1
        j += 1

    basis_ = [i for i in range(len(A[0]), len(new_matr_[0]))]

    xxy, basis_new = simplex(new_matr_, func_c, xy, basis_)
    if xxy == [] and basis_new == []:
        return []
    for i in range(len(A[0]), len(new_matr_[0])):
        if xxy[i] > 0:
            return False
    xk = [xxy[i] for i in range(len(A[0]))]

    N_Np = [i for i in range(len(xk)) if xk[i] == 0]
    print(N_Np)
    bas_ = []
    j = 0
    for i in basis_new:
        if i in N_Np:
            N_Np.remove(i)
        if i in basis_:
            bas_.append(N_Np[j])
            j += 1
            continue
        bas_.append(i)

    new_A = matrix_by_id_col(A, bas_)
    det_nA = np.linalg.det(new_A)
    if det_nA == 0: #смена базиса
        for i in N_Np:
            if i in bas_:
                for j in N_Np:
                    if i != j:
                        bas_.remove(i)
                        bas_.append(j)
                    new_A = matrix_by_id_col(new_A, bas_)
                    det_nA = np.linalg.det(new_A)
                    if det_nA == 0:
                        break
    print(bas_)
    print("---ПОИСК ОПТИМАЛЬНОГО ВЕКТОРА---")
    xopt, bas = simplex(A, c, xk, bas_)
    return xopt



