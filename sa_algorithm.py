import os
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from itertools import combinations
from math import sqrt, exp
from multiprocessing import Pool
from random import choice, random, sample

from ortools.linear_solver import pywraplp
from utm import from_latlon
import pandas as pd
from time import time

def exact_solution(matrix, number_of_AEDs):
    solver = pywraplp.Solver.CreateSolver('CLP')

    # L is the number of Facilities (AEDs)
    L = len(matrix[0])

    # M is the number of Demands (OHCAs)
    M = len(matrix)

    # a is the binary variable if AED in j-th location is applied for i-th OHCA.
    a = [[solver.BoolVar('a' + str(i) + str(j)) for j in range(L)] for i in range(M)]

    # y is binary variable if AED is placed at i-th location.
    y = [solver.BoolVar('y' + str(i)) for i in range(L)]

    # Constraint: Place exact N number of AEDs.
    main_con = solver.Constraint(number_of_AEDs, number_of_AEDs)
    for i in range(L):
        main_con.SetCoefficient(y[i], 1)

    # Constraint: Every OHCA has AED to use.
    cons = []
    for i in range(M):
        cons.append(solver.Constraint(1, 1))
        for j in range(L):
            cons[i].SetCoefficient(a[i][j], 1)

    # Constraint: Use AEDs only from where it is placed.
    for i in range(M):
        s_cons = []
        for j in range(L):
            s_cons.append(solver.Constraint(-1, 0))
            s_cons[j].SetCoefficient(y[j], -1)
            s_cons[j].SetCoefficient(a[i][j], 1)

    objective = solver.Objective()
    for i in range(M):
        for j in range(L):
            objective.SetCoefficient(a[i][j], matrix[i][j])

    objective.SetMaximization()
    status = solver.Solve()
    if status in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        solution = []
        for i in range(L):
            solution.append(y[i].solution_value())
        return solution, objective.Value()
    else:
        return [], 0


def distance(facility_list, demand):
    sp_list = list()
    for facility in facility_list:
        candidateDistance = from_latlon(facility[0], facility[1])
        demandDistance = from_latlon(demand[0], demand[1])
        totalTime = sqrt(pow(abs(candidateDistance[0] - demandDistance[0]), 2) +
                         pow(abs(candidateDistance[1] - demandDistance[1]), 2)) / 50
        # Assuming average walking speed of Singaporean is 100m / min
        if totalTime < 1:
            sp_list.append(0.549 * pow(1, -0.584))
        elif totalTime > 15:
            sp_list.append(0)
        else:
            sp_list.append(0.549 * pow(totalTime, -0.584))

    return sp_list


def get_demand_list(demand):
    f = open('./OHCA_2010-2017/' + str(demand) + '.txt', "r")
    list_of_coord = f.read().split("\n")
    f.close()
    split_latlong = []
    for coord in list_of_coord:
        lat_long = coord.split("\t")
        split_latlong.append([float(i) for i in lat_long])

    return split_latlong


def get_facility_list(demand):
    f = open('./SubzonesPossibleLocations/' + str(demand) + '.txt', "r")
    list_of_coord = f.read().split("\n")
    f.close()
    split_latlong = []
    for coord in list_of_coord:
        lat_long = coord.split("\t")
        split_latlong.append([float(i) for i in lat_long])

    return split_latlong


def get_facility_coordinates(x_OV, solution_list, facility_list, number_of_aed):
    if x_OV == 0.0:
        return []
    final_facility_list = []
    special_facility_list = []
    for index, solution in enumerate(solution_list):
        if solution > 0.5:
            final_facility_list.append(facility_list[index])
        elif solution > 0:
            special_facility_list.append(facility_list[index])
    if len(final_facility_list) != number_of_aed:
        extra_number_needed = number_of_aed - len(final_facility_list)
        for samp in sample(special_facility_list, extra_number_needed):
            final_facility_list.append(samp)
    return final_facility_list


def get_distance_matrix(demand):
    demand_list = get_demand_list(demand)
    facility_list = get_facility_list(demand)
    final_matrix = []
    for each_demand in demand_list:
        final_matrix.append(distance(facility_list, each_demand))
    return final_matrix, facility_list


def get_weighted(objective_value, facility_dict, demand):
    percentage = facility_dict.get(demand)[0]
    number_of_ohca = facility_dict.get(demand)[2]
    return float(percentage) * objective_value / 100 / number_of_ohca


def get_demandNameList_and_facilityDict():
    facility_dict = dict()
    demand_name_list = list()
    total_number_of_AED = 9880

    # Generate random number of AEDs in each subzone. Generate random float between 0.0 to 1.0
    random_aed = list()
    for x in range(302):
        random_aed.append(random())
    # Still need to retrieve percentage and number of OHCAs
    aed_number = pd.read_csv('./number_of_aed.csv')
    x = 0
    for index, col in aed_number.iterrows():
        if col['OHCAs'] != 0:
            demand_name_list.append(col['Subzone'])
            percentage = col['Proportion'].replace('%', '')
            random_float = random_aed[x]
            number_of_AED = round(random_float / sum(random_aed) * total_number_of_AED)
            facility_dict[col['Subzone']] = [percentage, number_of_AED, col['OHCAs']]
            x += 1

    # Ensure that total is 9880.
    total = sum([value[1] for key, value in facility_dict.items()])
    if total > 9880:
        for i in range(total - 9880):
            random_demand = choice(list(facility_dict.keys()))
            if facility_dict.get(random_demand)[1] != 0:
                facility_dict[random_demand][1] = facility_dict.get(random_demand)[1] - 1
    elif total < 9880:
        for i in range(9880 - total):
            random_demand = choice(list(facility_dict.keys()))
            facility_dict[random_demand][1] = facility_dict.get(random_demand)[1] + 1
    return demand_name_list, facility_dict


def initial_placement(facility_dict, demand):
    # Return: return [demand, objective_value, facility_coordinates]
    if facility_dict.get(demand)[1] >= facility_dict.get(demand)[2]:
        number_of_aed = facility_dict.get(demand)[2]
    else:
        number_of_aed = facility_dict.get(demand)[1]
    file = str(demand) + "_" + str(number_of_aed) + ".txt"
    if file in os.listdir('./Solutions'):
        with open("./Solutions/" + file, 'r') as f:
            data = f.read().split("\n")
            o_v = float(data[0])
            if o_v == 0.0:
                return [demand, 0.0, []]
            coordinates = [d.split("\t") for d in data[1:]]
        return [demand, o_v, coordinates]
    else:
        # all the location with zero AEDs. Therefore, return 0.0.
        with open("./initial_solution_file_error.txt", 'a') as f:
            f.write("File not found for: " + str(file) + "\n")
        return [demand, 0.0, []]


def read_solution_file(file):
    if file in os.listdir('./Solutions'):
        with open("./Solutions/" + file, 'r') as f:
            data = f.read().split("\n")
            o_v = float(data[0])
            if o_v == 0.0:
                return [], 0.0
            coordinates = [d.split("\t") for d in data[1:]]
        return coordinates, o_v
    else:
        # all the location with zero AEDs. Therefore, return 0.0.
        with open("./solution_file_error.txt", 'a') as f:
            f.write("File not found for: " + str(file) + "\n")
        return [], 0.0


def calculate_new(facility_dict, ov_dict, facility_list_dict):
    new_facility_dict = facility_dict.copy()
    new_ov_dict = ov_dict.copy()
    new_facility_list_dict = facility_list_dict.copy()

    # Randomly select 2 locations to swap AED, and they cannot be the same location.
    while True:
        x = choice(list(ov_dict.keys()))
        y = choice(list(ov_dict.keys()))
        if x != y:
            break

    initial_ov_xy = ov_dict.get(x) + ov_dict.get(y)

    # If number of AED in the subzone is 0, no movements is possible.
    if facility_dict.get(y)[1] and facility_dict.get(x)[1] == 0:
        return False, facility_dict, ov_dict, facility_list_dict
    elif facility_dict.get(x)[1] == 0:
        aed_combination = [[facility_dict.get(x)[1], facility_dict.get(y)[1]],
                           [facility_dict.get(x)[1] + 1, facility_dict.get(y)[1] - 1]]
    elif facility_dict.get(y)[1] == 0:
        aed_combination = [[facility_dict.get(x)[1] - 1, facility_dict.get(y)[1] + 1],
                           [facility_dict.get(x)[1], facility_dict.get(y)[1]]]
    else:
        aed_combination = [[facility_dict.get(x)[1] - 1, facility_dict.get(y)[1] + 1],
                           [facility_dict.get(x)[1] + 1, facility_dict.get(y)[1] - 1]]

    # if number of AEDs > number of OHCA, just read the file for max number of OHCA
    if aed_combination[0][0] >= facility_dict.get(x)[2]:
        x_first_file = str(x) + "_" + str(facility_dict.get(x)[2]) + ".txt"
    else:
        x_first_file = str(x) + "_" + str(aed_combination[0][0]) + ".txt"
    if aed_combination[0][1] >= facility_dict.get(y)[2]:
        y_first_file = str(y) + "_" + str(facility_dict.get(y)[2]) + ".txt"
    else:
        y_first_file = str(y) + "_" + str(aed_combination[0][1]) + ".txt"
    if aed_combination[1][0] >= facility_dict.get(x)[2]:
        x_second_file = str(x) + "_" + str(facility_dict.get(x)[2]) + ".txt"
    else:
        x_second_file = str(x) + "_" + str(aed_combination[1][0]) + ".txt"
    if aed_combination[1][1] >= facility_dict.get(y)[2]:
        y_second_file = str(y) + "_" + str(facility_dict.get(y)[2]) + ".txt"
    else:
        y_second_file = str(y) + "_" + str(aed_combination[1][1]) + ".txt"

    x_first_file_facility_list, x_first_file_ov = read_solution_file(x_first_file)

    y_first_file_facility_list, y_first_file_ov = read_solution_file(y_first_file)

    x_second_file_facility_list, x_second_file_ov = read_solution_file(x_second_file)

    y_second_file_facility_list, y_second_file_ov = read_solution_file(y_second_file)

    ov_diff = max(
        [initial_ov_xy, x_first_file_ov + y_first_file_ov, x_second_file_ov + y_second_file_ov]) - initial_ov_xy

    # Return: check_True, new_facility_dict, new_ov_dict, new_facility_list_dict
    if ov_diff == 0:
        return False, facility_dict, ov_dict, facility_list_dict

    elif ov_diff == x_first_file_ov + y_first_file_ov - initial_ov_xy:
        new_facility_dict[x][1] = aed_combination[0][0]
        new_facility_dict[y][1] = aed_combination[0][1]
        new_ov_dict[x] = x_first_file_ov
        new_ov_dict[y] = y_first_file_ov
        new_facility_list_dict[x] = x_first_file_facility_list
        new_facility_list_dict[y] = y_first_file_facility_list
        return True, new_facility_dict, new_ov_dict, new_facility_list_dict
    else:
        new_facility_dict[x][1] = aed_combination[1][0]
        new_facility_dict[y][1] = aed_combination[1][1]
        new_ov_dict[x] = x_second_file_ov
        new_ov_dict[y] = y_second_file_ov
        new_facility_list_dict[x] = x_second_file_facility_list
        new_facility_list_dict[y] = y_second_file_facility_list
        return True, new_facility_dict, new_ov_dict, new_facility_list_dict


def parallel_sa(variables):
    start_time = time()
    # variables: temperature, cooling_rate, max_iter, tol_value
    temperature = variables[0]
    cooling_rate = variables[1]
    maximum_iteration = 20
    tolerance_value = 0.001
    file_name = "SA_" + str(temperature) + "_" + str(cooling_rate) + "_"

    # facility_dict -> demand: [percentage, no_of_AED, no_of_ohca]
    demand_namelist, facility_dict = get_demandNameList_and_facilityDict()

    executor = Pool()
    ov_dict = dict()
    facility_list_dict = dict()
    # Getting OV from all Subzones
    # Return: return [demand, objective_value, facility_coordinates]
    func = partial(initial_placement, facility_dict)
    for result in executor.map(func, demand_namelist):
        ov_dict[result[0]] = result[1]
        facility_list_dict[result[0]] = result[2]
    executor.close()
    executor.join()

    print("Done with initial ov calculations.")
    current_ov = sum(ov_dict.values())
    count = 0
    index = 1
    while True:
        updated = False
        # check_True tells us if new_ov >  current_ov
        check_True, new_facility_dict, new_ov_dict, new_facility_list_dict = calculate_new(facility_dict, ov_dict, facility_list_dict)
        count += 1
        if check_True:
            count = 0
            facility_dict, ov_dict, facility_list_dict = new_facility_dict, new_ov_dict, new_facility_list_dict
            current_ov = sum(ov_dict.values())
            updated = True
        else:
            new_ov = sum(new_ov_dict.values())
            if count > maximum_iteration:
                temperature = temperature * cooling_rate
                count = 0
                if temperature < tolerance_value:
                    print("Done with " + file_name)
                    total_time_taken = time() - start_time
                    with open("./" + file_name + "iter_ov.txt", 'a') as f:
                        f.write("Total time taken: " + str(total_time_taken))
                    break
                continue
            else:
                # If same, current_ov - new_ov == 0, y will never be smaller. So just skip.
                if current_ov == new_ov:
                    continue
                else:
                    y = random()
                    if y < exp((current_ov - new_ov) / temperature):
                        facility_dict, ov_dict, facility_list_dict = new_facility_dict, new_ov_dict, new_facility_list_dict
                        current_ov = sum(ov_dict.values())
                        updated = True

        # Create new file every time there's a new update.
        if updated:
            with open("./sa_results/" + file_name + "iteration_" + str(index) + ".txt", 'a') as file:
                for facil_coords in list(facility_list_dict.values()):
                    for each_coords in facil_coords:
                        file.write(str(each_coords[0]) + "\t" + str(each_coords[1]) + "\n")
                for k, v in facility_dict.items():
                    file.write(str(k) + "\t")
                    for value in v:
                        file.write(str(value) + "\t")
                    file.write("\n")
            with open("./" + file_name + "iter_ov.txt", 'a') as f:
                f.write(str(sum(ov_dict.values())) + "\t" + str(index) + "\n")
            index += 1
        else:
            continue


def sa_algorithm():
    # Initialise variables in list [[temp, cool_rate]]
    variable_list = [[40, 0.5], [40, 0.6], [40, 0.7], [60, 0.5], [60, 0.6], [60, 0.7]]
    executor = ThreadPoolExecutor()
    result = executor.map(parallel_sa, variable_list)


sa_algorithm()