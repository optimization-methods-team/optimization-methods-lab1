import numpy as np
from matrix import matrix_by_id_col, gaussMeth, determinant, mult_vec

EPS = 0.000000001


def combination(array, num):
    """Рекурсивная функция поиска различных комбинаций из списка array и количеством элементов num"""
    if num == 0:
        return [[]]
    list_ = []  # возвращаемый список комбинаций

    for j in range(0, len(array)):
        emptyArray = array[j]  # убираем элемент и составляем без него комбинации (потом его будем добавлять обратно
        recurList = array[j + 1:]
        for x in combination(recurList, num-1):
            list_.append([emptyArray] + x)
    return list_


def get_basis_matrs(A: list):
    """Генерирование матриц размера M*M с ненулевым определителем"""
    M = len(A)  # количество строк в матрице
    N = len(A[0])  # количество столбцов в матрице

    basis_matrs = []
    basis_combinations_indexes = []
    all_indexes = [i for i in range(N)]

    for list_col in combination(all_indexes, M):
        basis_matr = matrix_by_id_col(A, list_col)
        det = determinant(basis_matr)
        if abs(det) > EPS:  # проверяем, что определитель отличен от нуля
            basis_matrs.append(basis_matr)  # получаем все такие матрицы и индексы комбинаций записываем
            basis_combinations_indexes.append(list_col)

    print("Количество базисных матриц: ", len(basis_matrs))

    return basis_matrs, basis_combinations_indexes


def validation_vec(solve) -> bool:
    """Проверка допустимости вектора"""
    for value in solve:
        if value < 0 or value > 1e+15:
            return False
    return True


def get_all_possible_vectors(A: list, b: list):
    """Получение допустимых опорных векторов"""

    N = len(A[0])
    M = len(A)
    vectors = []

    if M >= N:  # Рассматривается матрица A[M,N}, где число строк меньше числа столбцов (M < N)
        return vectors
    else:
        basis_matrs, basis_combinations_indexes = get_basis_matrs(A)

    for i in range(len(basis_matrs)):  # Для всех матриц с ненулевым определителем

        solve = gaussMeth(basis_matrs[i], b)  # Решаем систему вида A[M,N_k]*x[N_k]=b[M]
        if not validation_vec(solve):
            continue

        vec = [0 for i in range(N)]  # Дополняем нулями до N
        k = 0
        for j in basis_combinations_indexes[i]:
            vec[j] = solve[k]
            k += 1
        vectors.append(vec)
    return vectors



def solve_brute_force(A: list, b: list, c: list):
    """Поиск наименьшего значения функции цели"""
    vectors = get_all_possible_vectors(A, b)  # получаем все возможные опорные вектора
    if len(vectors) == 0:  # если их нет, нет оптимального решения
        return []

    solution = vectors[0]
    target_min = mult_vec(solution, c)

    for vec in vectors:
        value_func = mult_vec(vec, c)
        if value_func < target_min:  # находим минимум
            target_min = value_func  # значение функции цели в крайней точке
            solution = vec
            print("Лучшее значение целевой функции: ", target_min)

    return solution

