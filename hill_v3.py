import os
from concurrent.futures.process import ProcessPoolExecutor
from functools import partial
from itertools import combinations
from math import sqrt
from multiprocessing import Pool

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


def get_facility_coordinates(solution_list, facility_list):
    final_facility_list = []
    for index, solution in enumerate(solution_list):
        if solution == 1.0:
            final_facility_list.append(facility_list[index])
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
    aed_number = pd.read_csv('./number_of_aed.csv')
    for index, col in aed_number.iterrows():
        if col['OHCAs'] != 0:
            demand_name_list.append(col['Subzone'])
            percentage = col['Proportion'].replace('%', '')
            facility_dict[col['Subzone']] = [percentage, col['AED'], col['OHCAs']]

    return demand_name_list, facility_dict


def parallel_distance(facility_coords_dict_values, demand):
    sp_list = list()
    for facility_list in facility_coords_dict_values:
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


def initial_placement(facility_dict, demand):
    distance_matrix, facility_list = get_distance_matrix(demand)
    solution_list, objective_value = exact_solution(distance_matrix, facility_dict.get(demand)[1])
    list_of_facilities_coordinates = get_facility_coordinates(solution_list, facility_list)
    weighted_average = get_weighted(objective_value, facility_dict, demand)
    return [demand, list_of_facilities_coordinates, weighted_average]


# def parallel_process(x, y, facility_dict, x_distance_matrix, x_facility_list, y_distance_matrix, y_facility_list, aed_combination):
#     x_solution, x_OV = exact_solution(x_distance_matrix, aed_combination[0])
#     list_of_x_facilities_coordinates = get_facility_coordinates(x_solution, x_facility_list)
#     weighted_average_x = float(facility_dict.get(x)[0]) * x_OV / 100 / facility_dict.get(x)[2]
#     y_solution, y_OV = exact_solution(y_distance_matrix, aed_combination[1])
#     list_of_y_facilities_coordinates = get_facility_coordinates(y_solution, y_facility_list)
#     weighted_average_y = float(facility_dict.get(y)[0]) * y_OV / 100 / facility_dict.get(y)[2]
#     return [weighted_average_x, weighted_average_y, list_of_x_facilities_coordinates, list_of_y_facilities_coordinates, aed_combination]


def calculate_ov_facility(x, facility_dict, number_of_aed):
    x_distance_matrix, x_facility_list = get_distance_matrix(x)
    x_solution, x_OV = exact_solution(x_distance_matrix, number_of_aed)
    list_of_x_facilities_coordinates = get_facility_coordinates(x_solution, x_facility_list)
    weighted_average_x = float(float(facility_dict.get(x)[0]) * x_OV / 100 / facility_dict.get(x)[2])
    with open('./Solutions/'+str(x)+"_"+str(number_of_aed)+".txt", 'w') as f:
        f.write(str(weighted_average_x))
        for each_coords in list_of_x_facilities_coordinates:
            f.write("\n")
            f.write(str(each_coords[0]) + "\t" + str(each_coords[1]))
    return list_of_x_facilities_coordinates, weighted_average_x


def check_if_available(file):
    try:
        if file in os.listdir('./Solutions'):
            with open("./Solutions/" + file, 'r') as f:
                data = f.read().split("\n")
                o_v = float(data[0])
                coordinates = [d.split("\t") for d in data[1:]]
            return coordinates, o_v
        return None
    except ValueError:
        return None


def move_aed_parallel(facility_dict, ov_dict, x_y):
    x = x_y[0]
    y = x_y[1]
    initial_ov_xy = ov_dict.get(x) + ov_dict.get(y)

    # If number of AED in the subzone is 0, no movements is possible.
    if facility_dict.get(y)[1] and facility_dict.get(x)[1] == 0:
        return 0, x, y
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

    if check_if_available(x_first_file) is None:
        x_first_file_facility_list, x_first_file_ov = calculate_ov_facility(x, facility_dict, aed_combination[0][0])
    else:
        x_first_file_facility_list, x_first_file_ov = check_if_available(x_first_file)

    if check_if_available(y_first_file) is None:
        y_first_file_facility_list, y_first_file_ov = calculate_ov_facility(y, facility_dict, aed_combination[0][1])
    else:
        y_first_file_facility_list, y_first_file_ov = check_if_available(y_first_file)

    if check_if_available(x_second_file) is None:
        x_second_file_facility_list, x_second_file_ov = calculate_ov_facility(x, facility_dict, aed_combination[1][0])
    else:
        x_second_file_facility_list, x_second_file_ov = check_if_available(x_second_file)

    if check_if_available(y_second_file) is None:
        y_second_file_facility_list, y_second_file_ov = calculate_ov_facility(y, facility_dict, aed_combination[1][1])
    else:
        y_second_file_facility_list, y_second_file_ov = check_if_available(y_second_file)

    ov_diff = max([initial_ov_xy, x_first_file_ov+y_first_file_ov, x_second_file_ov+y_second_file_ov]) - initial_ov_xy
    if ov_diff == 0:
        return 0, x, y

    elif ov_diff == x_first_file_ov+y_first_file_ov-initial_ov_xy:
        return ov_diff, x_first_file_ov, y_first_file_ov, x_first_file_facility_list, y_first_file_facility_list, \
               aed_combination[0][0], aed_combination[0][1], x, y
    else:
        return ov_diff, x_second_file_ov, y_second_file_ov, x_second_file_facility_list, y_second_file_facility_list, \
               aed_combination[1][0], aed_combination[1][1], x, y


def hill_v3():
    # Initialize AED assignment to subzones(according to what Reza suggested);
    # facility_dict -> subzone: [percentage, number of AEDs, number of OHCAs]
    demand_namelist, facility_dict = get_demandNameList_and_facilityDict()
    func = partial(initial_placement, facility_dict)
    executor = ProcessPoolExecutor()
    ov_dict = dict()
    facility_list_dict = dict()

    # Getting OV from all Subzones
    # Return: return [demand, list_of_facilities_coordinates, objective_value]
    for result in executor.map(func, demand_namelist):
        ov_dict[result[0]] = result[2]
        facility_list_dict[result[0]] = result[1]
    executor.shutdown()
    current_OV = sum(ov_dict.values())
    with open("./iter_ov.txt", 'a') as f:
        f.write(str(current_OV) + "\t 0 \n")
    # Do the following
    # Do the following in Parallel for each pair of subzones
    # Move one AED from one subzone to the other;
    # Re - run AED placement for these two subzones;
    # Re - calculate overall objective function(using new placement of these two subzones and the existing placement for other subzones);

    iteration = 1
    total_combinations = [[x, y] for x,y in combinations(demand_namelist, 2)]
    while True:
        start_time = time()
        func = partial(move_aed_parallel, facility_dict, ov_dict)
        max_ov_diff = 0
        updated = False
        with Pool() as p:
            res = p.map(func, total_combinations)
            for result in res:
                if result[0] > max_ov_diff:
                    updated = True
                    max_ov_diff = result[0]
                    max_x_OV = result[1]
                    max_y_OV = result[2]
                    max_list_x = result[3]
                    max_list_y = result[4]
                    max_aed_combi = [result[5], result[6]]
                    max_x = result[7]
                    max_y = result[8]
            # Return: ov_diff, x_second_file_ov, y_second_file_ov, x_second_file_facility_list, y_second_file_facility_list, \
            #                aed_combination[1][0], aed_combination[1][1], x, y

            if updated:
                ov_dict[max_x] = max_x_OV
                ov_dict[max_y] = max_y_OV
                facility_list_dict[max_x] = max_list_x
                facility_list_dict[max_y] = max_list_y
                facility_dict.get(max_x)[1] = max_aed_combi[0]
                facility_dict.get(max_y)[1] = max_aed_combi[1]
                with open("./results/iteration" + str(iteration) + ".txt", 'a') as file:
                    for facil_coords in list(facility_list_dict.values()):
                        for each_coords in facil_coords:
                            file.write(str(each_coords[0]) + "\t" + str(each_coords[1]) + "\n")
                    for k, v in facility_dict.items():
                        file.write(str(k) + ": " + str(v) + "\n")
                end_time = time()
                with open("./iter_ov.txt", 'a') as f:
                    f.write(str(sum(ov_dict.values())) + "\t" + str(iteration) + "\t" + str(end_time-start_time) + "\n")

            else:
                print("No better solution found.")
                break

        iteration += 1
        # Find the AED allocation and placement with the best overall objective value;
        # Set delta to be the difference between the best overall objective value of the current iteration and best overall objective value obtained from the last iteration;
        # Until delta < a small threshold value

hill_v3()
