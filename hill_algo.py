from concurrent.futures.process import ProcessPoolExecutor
from functools import partial
from itertools import combinations
from math import sqrt
from multiprocessing import Pool

from ortools.linear_solver import pywraplp
from utm import from_latlon
import pandas as pd


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


def get_max(facility_list, demand):
    demand_list = get_demand_list(demand)
    total_OV = 0
    for each_demand in demand_list:
        distances = distance(facility_list, each_demand)
        current_max = 0
        for i in distances:
            if i > current_max:
                current_max = i
        total_OV += current_max
    return total_OV / len(demand_list)


def get_weighted(objective_value, facility_dict, demand):
    percentage = facility_dict.get(demand)[0]
    number_of_aed = facility_dict.get(demand)[1]
    return float(percentage) * objective_value / 100 / number_of_aed


def get_demandNameList_and_facilityDict():
    facility_dict = dict()
    demand_name_list = list()
    aed_number = pd.read_csv('./number_of_aed.csv')
    for index, col in aed_number.iterrows():
        if col['OHCAs'] != 0:
            demand_name_list.append(col['Subzone'])
            percentage = col['Proportion'].replace('%', '')
            facility_dict[col['Subzone']] = [percentage, col['AED']]
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


def parallel_process(x_percentage, x_distance_matrix, x_facility_list, y_percentage, y_distance_matrix, y_facility_list,
                     aed_combination):
    x_solution, x_OV = exact_solution(x_distance_matrix, aed_combination[0])
    list_of_x_facilities_coordinates = get_facility_coordinates(x_solution, x_facility_list)
    weighted_average_x = float(x_percentage) * x_OV / 100 / aed_combination[0]
    y_solution, y_OV = exact_solution(y_distance_matrix, aed_combination[1])
    list_of_y_facilities_coordinates = get_facility_coordinates(y_solution, y_facility_list)
    weighted_average_y = float(y_percentage) * y_OV / 100 / aed_combination[1]
    return [weighted_average_x, weighted_average_y, list_of_x_facilities_coordinates, list_of_y_facilities_coordinates, aed_combination]


def hill_algo():
    # Initialize AED assignment to subzones(according to what Reza suggested);
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

    index = 1

    for x, y in combinations(demand_namelist, 2):
        print(x)
        print(y)
        total_aed = facility_dict.get(x)[1] + facility_dict.get(y)[1]
        x_percentage = facility_dict.get(x)[0]
        y_percentage = facility_dict.get(y)[0]
        x_distance_matrix, x_facility_list = get_distance_matrix(x)
        print("Matrix done for x: ", x)
        y_distance_matrix, y_facility_list = get_distance_matrix(y)
        print("Matrix done for y: ", y)
        aed_combination = [[i + 1, total_aed - (i + 1)] for i in range(total_aed - 1)]
        print("AED combi done for: " + x + " and " + y)
        initial_ov_xy = ov_dict.get(x) + ov_dict.get(y)
        updated = False
        func = partial(parallel_process, x_percentage, x_distance_matrix, x_facility_list, y_percentage,
                       y_distance_matrix, y_facility_list)
        # Return: [x_OV, y_OV, list_of_x_facilities_coordinates, list_of_y_facilities_coordinates, aed_combination]
        # executor = ProcessPoolExecutor()
        # for result in executor.map(func, aed_combination):
        #     if result[0] != 0 and result[1] != 0 and result[0] + result[1] > initial_ov_xy:
        #         ov_dict[x] = result[0]
        #         ov_dict[y] = result[1]
        #         facility_list_dict[x] = result[2]
        #         facility_list_dict[y] = result[3]
        #         initial_ov_xy = ov_dict.get(x) + ov_dict.get(y)
        #         facility_dict.get(x)[1] = result[4][0]
        #         facility_dict.get(y)[1] = result[4][1]
        #         updated = True
        # executor.shutdown()

        with Pool() as p:
            res = p.map(func, aed_combination)
            for result in res:
                print("Done for AED combi: " + str(result[4][0]) + " and " + str(result[4][1]))
                if result[0] != 0 and result[1] != 0 and result[0] + result[1] > initial_ov_xy:
                    ov_dict[x] = result[0]
                    ov_dict[y] = result[1]
                    facility_list_dict[x] = result[2]
                    facility_list_dict[y] = result[3]
                    initial_ov_xy = ov_dict.get(x) + ov_dict.get(y)
                    facility_dict.get(x)[1] = result[4][0]
                    facility_dict.get(y)[1] = result[4][1]
                    updated = True
        # Re-calculate max_OV in all subzones
        if updated:
            with open("./results/iteration" + str(index) + ".txt", 'a') as file:
                for facil_coords in list(facility_list_dict.values()):
                    for each_coords in facil_coords:
                        file.write(str(each_coords[0]) + "\t" + str(each_coords[1]) + "\n")
                file.write(str(sum(ov_dict.values())) + "\n")
                for k, v in facility_dict.items():
                    file.write(str(k) + ": " + str(v) + "\n")

        with open("./hill_algo_log.txt", 'a') as f:
            f.write("Done " + str(index) + "\n")
        with open("./iter_ov.txt", 'a') as f:
            f.write(str(sum(ov_dict.values())) + "\t" + str(index) + "\n")

        index += 1
        # Find the AED allocation and placement with the best overall objective value;
        # Set delta to be the difference between the best overall objective value of the current iteration and best overall objective value obtained from the last iteration;
        # Until delta < a small threshold value


hill_algo()
