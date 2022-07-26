import copy
import math
import sys
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

# 简单模型

oneway_efficiency = 0.8  # 单行通行的效率因子，值越小，单向通行效率越高

size = 4  # 道路有向图的顶点数目

road_data = [
    [0,   0,   3,   3],  # 二维数组中道路数据的行序号
    [1,   2,   1,   2],  # 二维数组中道路数据的列序号
    [10, 200, 200, 10],  # 上述行和列对应的道路数据，为双向通行时的cost，
    [2,   2,   2,   2]   # 1 单向，2 双向
]

traffic_data = [
    [0,  2, 1,  3],   # 二维数组中交通流量的行序号
    [2,  0, 3,  1],   # 二维数组中交通流量的列序号
    [20, 5, 10, 10],  # 上述行和列对应的流量数据，为通行车辆数
]

'''
# 五角场环岛

oneway_efficiency = 0.8

size = 10

road_data = [
    [0,   1,   2,   3,   4,   5,   6,   7,   8,   9],
    [5,   6,   7,   8,   9,   6,   7,   8,   9,   5],
    [100, 100, 100, 100, 100, 200, 200, 200, 200, 200],  
    [2,   2,   2,   2,   2,   2,   2,   2,   2,   2] 
]

traffic_data = [
    [0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,  4,  4,  4,  4],
    [1,  2,  3,  4,  2,  3,  4,  0,  3,  4,  0,  1,  4,  0,  1,  2,  0,  1,  2,  3],
    [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
]
'''

'''
# 五角场 <-> 复附 扇形区域

oneway_efficiency = 0.7

size = 26

road_data = [
    [0,   1,   5,   8,   14,  6,   9,   15,  2,   7,   10,  16,  11,  17,  3,   12,  24,  0,   4,   13,  19,  1,   2,   3,   5,   6,   8,   9,   13,  12,  11,  14,  15,  16,  17,  18,  20,  21,  22,  23,  24],
    [1,   5,   8,   14,  20,  9,   15,  21,  7,   10,  16,  22,  17,  23,  12,  18,  18,  4,   13,  19,  25,  2,   3,   4,   6,   7,   9,   10,  12,  11,  10,  15,  16,  17,  18,  19,  21,  22,  23,  24,  25],
    [250, 100, 100, 200, 200, 300, 400, 400, 200, 220, 250, 250, 400, 400, 300, 300, 300, 250, 200, 200, 200, 100, 100, 100, 200, 200, 250, 250, 200, 250, 250, 350, 250, 250, 250, 250, 400, 400, 400, 400, 400],
    [2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   1,   1,   1,   1,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   2,   1,   1,   1,   2,   2,   2,   2,   2,   2,   2,   1,   1,   1],
]

traffic_data = [
    [0,   20,  0,   25,  1,  4,  8,  13, 14, 19, 20, 25, 13, 19, 23, 2,  24, 2,  6,  2,  15, 22,  9,  9,  20, 0,   22, 17, 16, 18, 17, 15, 13, 9,  2,  10, 2,  11, 13, 15, 19, 24, 2,  21, 6,  21, 13],
    [20,  0,   25,  0,   4,  1,  13, 8,  19, 14, 25, 20, 22, 22, 2,  23, 2,  24, 2,  6,  2,  9,   22, 20, 9,  25,  25, 3,  4,  0,  19, 13, 15, 2,  9,  2,  10, 13, 10, 19, 15, 2,  24, 6,  21, 25, 20],
    [100, 100, 100, 100, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 10, 10, 10, 10, 10, 10, 10, 10,  10, 10, 10, 100, 50, 5,  5,  5,  10, 10, 10, 10, 10, 5,  5,  10, 15, 10, 10, 5,  5,  15, 10, 10, 10],
]
'''


def calc_total_traffic_cost(traffic_costs):
    total_traffic_cost = 0
    for row in range(size):
        for column in range(size):
            traffic_cost = traffic_costs[row][column]
            if math.isinf(traffic_cost):
                return math.inf, (row, column)
            else:
                total_traffic_cost += traffic_cost
    return total_traffic_cost, (-1, -1)


def calc_traffic_costs(road_matrix, predecessors, traffic_matrix):
    traffic_costs = []
    for row in range(size):
        columns = []
        for column in range(size):
            columns.append(0)
        traffic_costs.append(columns)

    for row in range(size):
        for column in range(size):
            traffic = traffic_matrix[row][column]
            if traffic != 0:
                current = column
                predecessor = predecessors[row][current]
                while predecessor >= 0:
                    traffic_costs[predecessor][current] += traffic * road_matrix[predecessor][current]
                    current = predecessor
                    predecessor = predecessors[row][current]

                if current != row:
                    traffic_costs[row][column] = math.inf

    return traffic_costs


def make_oneway(road_matrix, row, column):
    cost = road_matrix[row][column]
    reverse_cost = road_matrix[column][row]
    if cost != 0 and reverse_cost != 0:
        optimized_road_matrix = copy.deepcopy(road_matrix)
        optimized_road_matrix[row][column] = (cost + reverse_cost) / 2 * oneway_efficiency
        optimized_road_matrix[column][row] = 0
        return optimized_road_matrix


def optimize_roads(road_matrix, traffic_matrix, traffic_costs, total_traffic_cost):
    roads = []
    for row in range(size):
        for column in range(size):
            if road_matrix[row][column] != 0 and road_matrix[column][row] != 0:
                roads.append((row, column))

    def get_oneway_gain(current_road):
        cost = traffic_costs[current_road[0]][current_road[1]]
        reverse_cost = traffic_costs[current_road[1]][current_road[0]]
        if cost == 0 and reverse_cost == 0:
            return 1
        elif reverse_cost == 0:
            return math.inf
        else:
            return cost / reverse_cost

    roads.sort(key=get_oneway_gain, reverse=True)

    for road in roads:
        row = road[0]
        column = road[1]
        optimized_road_matrix = make_oneway(road_matrix, row, column)
        if optimized_road_matrix is not None:
            optimized_predecessors = \
                shortest_path(csgraph=csr_matrix(optimized_road_matrix), directed=True, return_predecessors=True)[1]
            optimized_traffic_costs = calc_traffic_costs(optimized_road_matrix, optimized_predecessors, traffic_matrix)
            optimized_total_traffic_cost = calc_total_traffic_cost(optimized_traffic_costs)[0]

            if optimized_total_traffic_cost < total_traffic_cost:
                road_matrix = optimize_roads(optimized_road_matrix, traffic_matrix, optimized_traffic_costs,
                                             optimized_total_traffic_cost)
                break
            else:
                more_optimized = False
                current = row
                predecessor = optimized_predecessors[column][current]
                while predecessor >= 0:
                    more_optimized_road_matrix = make_oneway(optimized_road_matrix, predecessor, current)
                    if more_optimized_road_matrix is not None:
                        optimized_road_matrix = more_optimized_road_matrix
                        more_optimized = True
                    current = predecessor
                    predecessor = optimized_predecessors[column][current]
                if more_optimized:
                    optimized_predecessors = \
                        shortest_path(csgraph=csr_matrix(optimized_road_matrix), directed=True,
                                      return_predecessors=True)[1]
                    optimized_traffic_costs = \
                        calc_traffic_costs(optimized_road_matrix, optimized_predecessors, traffic_matrix)
                    optimized_total_traffic_cost = calc_total_traffic_cost(optimized_traffic_costs)[0]
                    if optimized_total_traffic_cost < total_traffic_cost:
                        road_matrix = optimize_roads(optimized_road_matrix, traffic_matrix, optimized_traffic_costs,
                                                     optimized_total_traffic_cost)
                        break

    return road_matrix


def run():
    for index in range(len(road_data[3])):
        if road_data[3][index] == 2:
            road_data[0].append(road_data[1][index])
            road_data[1].append(road_data[0][index])
            road_data[2].append(road_data[2][index])
        else:
            road_data[2][index] = road_data[2][index] * oneway_efficiency

    road_matrix = csr_matrix((road_data[2], (road_data[0], road_data[1])), shape=(size, size)).toarray()

    traffic_matrix = csr_matrix((traffic_data[2], (traffic_data[0], traffic_data[1])), shape=(size, size)).toarray()

    for row in range(size):
        for column in range(size):
            if row == column and road_matrix[row][column] != 0:
                sys.exit("Invalid road matrix value at " + str(row) + ":" + str(column))

    for row in range(size):
        for column in range(size):
            if row == column and traffic_matrix[row][column] != 0:
                sys.exit("Invalid traffic matrix value at " + str(row) + ":" + str(column))

    predecessors = shortest_path(csgraph=csr_matrix(road_matrix), directed=True, return_predecessors=True)[1]
    traffic_costs = calc_traffic_costs(road_matrix, predecessors, traffic_matrix)
    total_traffic_cost, unroutable = calc_total_traffic_cost(traffic_costs)

    if total_traffic_cost == math.inf:
        sys.exit("No route from vertex " + str(unroutable[0]) + " to " + str(unroutable[1]))

    optimized_road_matrix = optimize_roads(road_matrix, traffic_matrix, traffic_costs, total_traffic_cost)

    optimized_predecessors = shortest_path(csgraph=csr_matrix(optimized_road_matrix), directed=True,
                                           return_predecessors=True)[1]
    optimized_traffic_costs = calc_traffic_costs(optimized_road_matrix, optimized_predecessors, traffic_matrix)
    optimized_total_traffic_cost = calc_total_traffic_cost(optimized_traffic_costs)[0]

    print("original total cost: " + str(total_traffic_cost))
    print("optimized total cost: " + str(optimized_total_traffic_cost))

    print("optimized oneway roads")
    for row in range(size):
        for column in range(size):
            if optimized_road_matrix[row][column] != 0 and optimized_road_matrix[column][row] == 0:
                print(str(row) + " --> " + str(column))


run()

