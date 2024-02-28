import copy
from matrix import transport_matrix
from meth_end_point_search import solve_brute_force, mult_vec
from simplex_method import find_start_bas
"""
Чтение файла. Сохраняем систему.
Предполагается, что в строке, начинающейся с goal_gunc, записана целевая функция.
В строке, начинающейся с idx, записаны индексы переменных, имеющих ограничение на знак >= 0.
"""


def read_file(filename):
    """Считывание функции цели, системы ограничений и ограничений на знак переменных"""
    system = []
    sign = []
    goal_func = []
    idx = []
    type_task = ''
    with open(filename, "r") as f:
        for line in f.readlines():
            expression = line.split()
            if expression[0] == "min" or expression[0] == "max":
                type_task = expression[0]
                continue

            if expression[0] == "goal_func":
                expression.remove("goal_func")
                for value in expression:
                    goal_func.append(float(value))
                continue

            if expression[0] == "idx":
                expression.remove("idx")
                for value in expression:
                    if int(value) > 0:
                        idx.append(int(value) - 1)  # пользователь не должен задумываться о том, что в программе
                        # индексация с 0, поэтому он записывает все индексы на один больше, чем мы предполагаем
                        continue
                    idx.append(int(value) + 1)  # если индекс пришел отрицательный,
                    # значит, что для него ограничение на знак <=0
                continue

            clean_data = []
            for value in expression:
                if value.isdigit() or value[0] == '-':
                    clean_data.append(float(value))
                else:
                    sign.append(value)
            system.append(clean_data)
    return system, sign, goal_func, idx, type_task


def to_canonical(system, sign, goal_func, idx, type_task: str):
    """Приведение к канонической форме"""
    # копирование данных, чтобы исходные остались прежними
    copy_sign = copy.deepcopy(sign)
    copy_system = copy.deepcopy(system)
    copy_idx = copy.deepcopy(idx)
    copy_goal_func = copy.deepcopy(goal_func)
    # приводим к канонической форме
    if type_task == "max":  # если задача максимизации, то нужно изменить коэффициенты функции цели на противоположные
        copy_goal_func = [-copy_goal_func[i] for i in range(len(copy_goal_func))]
    # сначала проверяем есть ли переменные с ограничением <=0
    for id_ in copy_idx:
        if id_ < 0:
            for i in range(len(copy_system)):
                copy_system[i][abs(id_)] = - copy_system[i][abs(id_)]
            copy_goal_func[abs(id_)] = - copy_goal_func[abs(id_)]

    copy_idx = [abs(copy_idx[i]) for i in range(len(copy_idx))]

    # потом переменные без ограничения на знак заменяем новыми
    # в том числе в ф-ии цели
    to_delete = []  # здесь будем хранить индексы "старых" переменных
    for i in range(len(copy_system[0]) - 1):
        if i not in copy_idx:
            # значит на знак нет ограничения
            for j in range(len(copy_system)):  # заменяем переменную без ограничения на u-v (разницу двух новых переменных)
                copy_system[j].insert(-1, copy_system[j][i])
                copy_system[j].insert(-1, -copy_system[j][i])
            copy_goal_func.append(copy_goal_func[i])
            copy_goal_func.append(-copy_goal_func[i])
            to_delete.append(i)

    to_delete = to_delete[::-1]

    for i in range(len(copy_system)):
        for j in to_delete:
            copy_system[i].pop(j)
    for j in to_delete:
        copy_goal_func.pop(j)

    # теперь заменяем все знаки на равенства
    for i in range(len(copy_system)):
        if copy_sign[i] == '<=':  # если знак <=
            for j in range(len(copy_system)):
                if j == i:
                    copy_system[j].insert(-1, 1.0)  # добавляем новую переменную с коэф-том 1
                    copy_idx.append(len(copy_system[j]) - 2)  # у переменной ограничение на знак
                else:
                    copy_system[j].insert(-1, 0.0)
            copy_goal_func.append(0.0)
            copy_sign[i] = '='  # делаем равенство
        if copy_sign[i] == '>=':  # если знак >=
            for j in range(len(copy_system)):
                if j == i:
                    copy_system[j].insert(-1, -1.0)  # добавляем новую переменную со коэф-том -1
                    copy_idx.append(len(copy_system[j]) - 2)
                else:
                    copy_system[j].insert(-1, 0.0)
            copy_goal_func.append(0.0)
            copy_sign[i] = '='

    # поменяем знак в A[i] и b[i], если b[i]<0
    for i in range(len(copy_system)):
        if copy_system[i][-1] < 0:
            copy_system[i] = [-copy_system[i][j] for j in range(len(copy_system[0]))]

    copy_idx = [i for i in range(len(copy_system[0]) - 1)]
    return copy_system, copy_sign, copy_goal_func, copy_idx


def direct_to_dual(system, sign, goal_func, idx, type_task):
    """Создание двойственной задачи по прямой"""
    dual_func = []
    dual_idx = []
    dual_sign = []
    for exp in system:
        dual_func.append(exp[-1])

    if type_task == "min":
        dual_type_task = "max"
    else:
        dual_type_task = "min"

    idx_l = [abs(id_) for id_ in idx if id_ < 0]  # x<=0
    idx_m = [id_ for id_ in idx if id_ >= 0]  # x>=0
    # создаем двойственную систему
    dual_system = transport_matrix(system)  # транспонированная матрица
    dual_system.pop(-1)
    for i in range(len(dual_system)):
        dual_system[i].append(goal_func[i])  # добавляем свободные члены
        if i in idx_m:  # и смотрим знаки новой системы: если на i было ограничение
            dual_sign.append('<=')  # то знак <=
        else:
            if i in idx_l:
                dual_sign.append('>=')  # то знак >=
            else:
                dual_sign.append('=')  # иначе =
        if sign[i] == '>=':
            dual_idx.append(i)
        if sign[i] == '<=':
            dual_idx.append(-i)  # если idx отрицательный, значит x[i] <= 0, если idx положит, то x[i] >= 0
    return dual_system, dual_sign, dual_func, dual_idx, dual_type_task



def getAb(system_):
    """ Вспомогательная ф-я для получения отдельной матрицы A и вектора b из system """
    A = []
    b = []
    for exp in system_:
        b.append(exp[-1])
        A.append(exp[:-1])
    return A, b


def print_system(system, sign, goal_func, idx, type_task):
    A, b = getAb(system)

    print('Целевая функция: ')
    for i in range(len(A[0])):
        if i != 0:
            print(' + ', end='')
        print(goal_func[i], ' * x[', i + 1, '] ', end='')
    print(' -> ', type_task)

    print('Ограничения: ')
    for i in range(len(A)):
        flag = 0
        for j in range(len(A[i])):
            if A[i][j] == 0:
                if j == 0:
                    flag = 1
                continue
            if j != 0 and flag != 1:
                print(' + ', end='')
            print(A[i][j], '*x[', j + 1, ']', end='', sep='')
            flag = 0

        print(' ', sign[i], b[i], sep=' ')

    print('Переменные с ограничением на знак: ')
    for i in idx:
        if i >= 0:
            print('x[', i + 1, ']>=0, ', end='')
            continue
        print('x[', abs(i) + 1, ']<=0, ', end='')
    print('\n')


system, sign, goal_func, idx, type_task = read_file("test2.txt")
print('---ИСХОДНАЯ ЗАДАЧА---')
print_system(system, sign, goal_func, idx, type_task)

print('---КАНОНИЧЕСКАЯ ФОРМА ПРЯМОЙ ЗАДАЧИ---')
system_c, sign_c, goal_func_c, idx_c = to_canonical(system, sign, goal_func, idx, type_task)
print_system(system_c, sign_c, goal_func_c, idx_c, type_task)

print('---ДВОЙСТВЕННАЯ ЗАДАЧА---')
system1, sign1, goal_func1, idx1, type_task1 = direct_to_dual(system, sign, goal_func, idx, type_task)
print_system(system1, sign1, goal_func1, idx1, type_task1)

print('---КАНОНИЧЕСКАЯ ФОРМА ДВОЙСТВЕННОЙ ЗАДАЧИ---')
system2, sign2, goal_func2, idx2 = to_canonical(system1, sign1, goal_func1, idx1, type_task1)
print_system(system2, sign2, goal_func2, idx2, "min")

print('---РЕШЕНИЕ ИСХОДНОЙ ЗАДАЧИ МЕТОДОМ ПЕРЕБОРА ОПОРНЫХ ВЕКТОРОВ---')
A, b = getAb(system_c)
solution = solve_brute_force(A, b, goal_func_c)
print('Вектор решения: ', solution)
print('', mult_vec(solution, goal_func_c))

print('---РЕШЕНИЕ ДВОЙСТВЕННОЙ ЗАДАЧИ МЕТОДОМ ПЕРЕБОРА ОПОРНЫХ ВЕКТОРОВ---')
A, b = getAb(system2)
solution = solve_brute_force(A, b, goal_func2)
print('Вектор решения: ', solution)
print('', mult_vec(solution, goal_func2))


print('---РЕШЕНИЕ ИСХОДНОЙ ЗАДАЧИ СИМПЛЕКС-МЕТОДОМ---')
A, b = getAb(system_c)
solution = find_start_bas(A, b, goal_func_c)
print('Вектор решения: ', solution)
print('', mult_vec(solution, goal_func_c))

print('---РЕШЕНИЕ ДВОЙСТВЕННОЙ ЗАДАЧИ СИМПЛЕКС-МЕТОДОМ---')
A, b = getAb(system2)
solution = find_start_bas(A, b, goal_func2)
print('Вектор решения: ', solution)
print('', mult_vec(solution, goal_func2))
