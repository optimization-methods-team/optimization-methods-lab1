import math
import numpy as np

def make_tableau(A, b, c):
    # Преобразование в таблицу симплекс метода:
    # __________________
    # |             |   |
    # |      A      | b |
    # |             |   |
    # -------------------
    # |      c      | 0 |
    # -------------------
    for i in range(len(c)):
        c[i] = -1 * c[i]

    xbasic = [equality + [x] for equality, x in zip(A, b)]
    # построение нижней строки таблицы
    z = c + [0]
    return xbasic + [z]

def can_optimized(tableau):
    # Проверяем, где можно увеличить неосновные значения, не уменьшая значение целевой функции.
    z = tableau[-1]
    print("Вектор целевой функции, ищем положительные коэффициенты")
    print(np.array(z[:-1]))
    return any(x > 0 for x in z[:-1])

def get_pivot_position(tableau):
    # Если значение целевой функции можно улучшить, мы ищем точку разворота.
    z = tableau[-1]
    column = 0
    # Правило Блэнда
    # найдем индекс первого положительного элемента
    bland = True
    if bland:
        for i in range(len(z) - 1):
            if z[i] > 0:
                # это будет индекс ведущего столбца
                column = i
                break

    restrictions = []

    # выбираем все элементы из столбца с индексом поворота
    # и почленно делим вектор свободных членов b на ведущий столбце
    for equality in tableau[:-1]:
        elem = equality[column]
        restrictions.append(math.inf if elem <= 0 else equality[-1] / elem)



    # Если все элементы ведущего столбца нули - задача неограничена
    if (all([r == math.inf for r in restrictions])):
        raise Exception("Linear program is unbounded.")
    # из полученного вектора извлекаем наименьший элемент
    # это будет индекс ведущей строки
    row = restrictions.index(min(restrictions))
    return row, column


def pivot_step(tableau, pivot_position):
    # зададим новую таблицу
    new_tableau = [[] for eq in tableau]

    # запишем индексы ведущего элемента
    i, j = pivot_position
    # запишем значение ведущего элемента
    pivot_value = tableau[i][j]
    # вычислим новую ведущую строку
    new_tableau[i] = np.array(tableau[i]) / pivot_value

    # делаем поворотный шаг и возвращаем новую таблицу
    for equality_i, equality in enumerate(tableau):
        # по методу Жордана-Гаусса изменяем все строки в таблице, кроме ведущей
        if equality_i != i:
            multiplier = np.array(new_tableau[i]) * tableau[equality_i][j]
            new_tableau[equality_i] = np.array(tableau[equality_i]) - multiplier
    return new_tableau


def is_basic(column):
    return sum(column) == 1 and len([c for c in column if c == 0]) == len(column) - 1


def get_solution(tableau):
    # извлекаем вектор решения из таблицы
    # а именно последнюю строку
    columns = np.array(tableau).T
    solutions = []
    for column in columns[:-1]:
        solution = 0
        if is_basic(column):
            one_index = column.tolist().index(1)
            solution = columns[-1][one_index]
        solutions.append(solution)

    return solutions


def simplex(c, A, b):
    np.set_printoptions(precision=2, suppress=True)

    iter = 0
    # построим таблицу исмплекс метода
    tableau = make_tableau(A, b, c)
    print("Инициализация таблицы:")
    print(np.array(tableau))

    # пока можем улучшать целевую функцию - делаем поворот
    while can_optimized(tableau):
        iter +=1
        # найдем ведущий элемент(индекс ведущего столбца и индекс ведущей строки)
        pivot_position = get_pivot_position(tableau)
        print("Индексы ведущего элемента")
        print(pivot_position)
        # выполним поворот
        # вычисление нового базисного решения через метод Жордана-Гаусса
        tableau = pivot_step(tableau, pivot_position)
        print("Таблица после замены базиса")
        print(np.array(tableau))

    print("Число итераций:")
    print(iter)
    return get_solution(tableau)