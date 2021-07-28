from collections import defaultdict, deque
import numpy as np
import csv
import ast


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(list)
        self.distances = {}

    def add_node(self, value):
        self.nodes.add(value)

    def add_edge(self, from_node, to_node, distance):
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.distances[(from_node, to_node)] = distance
        self.distances[(to_node, from_node)] = distance


def Dijsktra(graph, initial):
    visited = {initial: 0}
    path = {}
    nodes = set(graph.nodes)
    # Choose the smallest distance from the initial node
    while nodes:
        min_node = None
        for node in nodes:
            if node in visited:
                if min_node is None:
                    min_node = node
                elif visited[node] < visited[min_node]:
                    min_node = node
        if min_node is None:
            break
        nodes.remove(min_node)
        current_weight = visited[min_node]
        for edge in graph.edges[min_node]:
            weight = current_weight + graph.distances[(min_node, edge)]
            if edge not in visited or weight < visited[edge]:
                visited[edge] = weight
                path[edge] = min_node
    return visited, path


def PrintDistances(graph, initial):
    visited, path = Dijsktra(graph, initial)
    nodes = set(graph.nodes)
    for node in nodes:
        print("Distance node %s to node %s : %s" %(initial, node, visited[node]))
    return 0


def PrintPath(graph, initial, destination):
    visited, path = Dijsktra(graph, initial)
    road = [destination, path[destination]]
    if destination == initial:
        print("Take another destination")
    else:
        print("Distance node %s to node %s : %s" %(initial, destination, visited[destination]))
        while road[-1] != initial:
            road.append(path[road[-1]])
    print(road)
    return 0


def PlotGraphOneSourceInitNorm(graph, initial, nproj):
    visited, path = Dijsktra(graph, initial)
    destinations = []
    distances = []
    roadL = []
    for i in range(nproj):
        destinations.append(i)
        if i ==initial :
            road=[initial]
        else: 
            road = [initial,path[i]]
            while road[-1] != initial:
                road.append(path[road[-1]])
        roadL.append(len(road))
        distances.append(visited[i])
    return np.array(destinations), np.array(distances), np.array(roadL)


def PlotGraphOneSourceInit(graph, initial, nproj):
    visited, path = Dijsktra(graph, initial)
    destinations = []
    distances = []
    for i in range(nproj):
        destinations.append(i)
        distances.append(visited[i])
    return destinations, distances


def WriteCSVFile(DirName, PairsList, DistancesList):
    with open(DirName, 'w') as csvfile:
        fieldnames = ['Pairs', 'Distances']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        nproj = len(PairsList)
        for i in range(nproj):
            for j in range(len(PairsList[i])):
                writer.writerow({'Pairs': PairsList[i][j], 'Distances': DistancesList[i][j]})
    return 0


def ReadingCSVFile(DirName):
    ListOfPairs = []
    distances = []
    with open(DirName) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pair = row['Pairs']
            pair = ast.literal_eval(pair)
            ListOfPairs.append(pair)
            distances.append(np.float32(row['Distances']))
    return ListOfPairs,distances

def ComputeGraphAfterDCC(PairsList, DistancesList):
    G = Graph()
    nproj = len(PairsList)
    for i in range(nproj):
        G.add_node(i)
    for i in range(nproj):
        edges = G.edges[i]
        for j in range(len(PairsList[i])):
            if PairsList[i][j][1] not in edges:
                G.add_edge(PairsList[i][j][0], PairsList[i][j][1], DistancesList[i][j])
    return G


def ComputeGraphAfterReadingCSV(DirName):
    ListOfPairs, distances = ReadingCSVFile(DirName)
    G = Graph()
    for i in range(len(ListOfPairs)):
        nodes = G.nodes
        if ListOfPairs[i][0] not in nodes:
            G.add_node(ListOfPairs[i][0])
        if ListOfPairs[i][1] not in nodes:
            G.add_node(ListOfPairs[i][1])
        G.add_edge(ListOfPairs[i][0], ListOfPairs[i][1], distances[i])
    return G
