import sys
import time
from collections import defaultdict
from itertools import combinations
from pyspark import SparkConf, SparkContext
import copy
import json

start = time.time()


def to_list(a):
    return [a]


def append(a, b):
    a.append(b)
    return a


def extend(a, b):
    a.extend(b)
    return a


def create_graph(edge_list):
    graph_dictionary = defaultdict(list)
    for i, j in edge_list:
        graph_dictionary[i].append(j)
    return graph_dictionary


def girvan_newman(graph):
    def bfs_betweenness(root, g_dict):
        visit_level = [[root]]
        visited = [root]
        pos = 0
        while (visit_level[pos]):
            children_of_level = []
            for node in visit_level[pos]:
                children_of_level.extend(list(set(g_dict[node])))
            children_wo_repeats = list(set(children_of_level).difference(visited))

            if len(children_wo_repeats) == 0:
                break

            visit_level.append(children_wo_repeats)
            visited.extend(children_wo_repeats)
            pos += 1

        level = dict(enumerate(visit_level))
        # print(level)

        edge_set_bfs = []
        between_value = []
        no_of_levels = len(list(level.keys()))
        for i in range(no_of_levels - 1, 0, -1):
            key = i - 1
            # print(i)
            # print(key)
            parent_list = level[key]
            # print(parent_list)
            len_parent_list = len(parent_list)
            child_len = len(level[i])
            if len_parent_list == 1:
                parent = level[key]
                for j in level[i]:
                    edge_set_bfs.append((j, parent[0]))
                # print('case 1', edge_set_bfs)
            else:
                for j in level[i]:
                    parents = list(set(g_dict[j]) & set(list(parent_list)))
                    for parent in parents:
                        edge_set_bfs.append((j, parent))
                # print('case 2', edge_set_bfs)

        parent_dict_value = defaultdict(list)
        children_dict_value = defaultdict(list)
        for k, v in edge_set_bfs:
            parent_dict_value[k].append(v)
            for j in [v]:
                children_dict_value[j].append(k)

        # print("parent dict", parent_dict_value)
        # print("children dict", children_dict_value)

        non_root_nodes = list(parent_dict_value.keys())
        all_nodes = non_root_nodes.append(root)
        # print("non_roots", non_root_nodes)
        no_of_parents_children = {root: {"parent": parent_dict_value[root], "child": children_dict_value[root]}}
        # print(no_of_parents_children[root]['child'])

        node_value = {}
        leaves = []
        for i in non_root_nodes:
            child = children_dict_value[i]
            if len(child) == 0:
                leaves.append(i)
            no_of_parents_children[i] = {"parent": parent_dict_value[i], "child": child}
        # print("combined dict", no_of_parents_children)
        # print('leaves',leaves)

        shortest_paths = {root: 1}
        for j in level[1]:
            shortest_paths[j] = 1
        for i in range(2, no_of_levels):
            for k in level[i]:
                par = no_of_parents_children[k]["parent"]
                paths = 0
                for p in par:
                    paths = paths + shortest_paths[p]
                shortest_paths[k] = paths
        # print(shortest_paths)
        between_values = {}
        for i in edge_set_bfs:
            par = no_of_parents_children[i[0]]["parent"]
            if i[0] in leaves:
                between_values[i] = (1 * shortest_paths[i[1]]) / shortest_paths[i[0]]
            else:
                credit = 1
                for ch in no_of_parents_children[i[0]]["child"]:
                    credit = credit + between_values[(ch, i[0])]
                between_values[i] = (credit * shortest_paths[i[1]]) / shortest_paths[i[0]]

        betweenness_list = [(sorted(k), v) for k, v in between_values.items()]
        return betweenness_list

    bfs_all = []
    for i in graph.keys():
        bfs_all.extend(bfs_betweenness(i, graph))
    # print(bfs_all)

    final_betweenness = defaultdict(list)
    for k, v in bfs_all:
        final_betweenness[tuple(k)].append(v)
    # print(final_betweenness)

    final = []
    for i in final_betweenness:
        value = sum(final_betweenness[i]) / 2
        final.append((i, value))

    final = sorted(final, key=lambda x: (-x[1], x[0][0]))
    return final


def edge_removal(graph, between_list):
    highest_betweenness = between_list[0][1]

    edges_to_remove = []
    for i in betweenness:
        if i[1] == highest_betweenness:
            edges_to_remove.append(i[0])
        else:
            break

    for i in edges_to_remove:
        # print("edge removed:", i)
        graph[i[0]].remove(i[1])
        graph[i[1]].remove(i[0])
    return graph


def check_connected(cgraph, v):
    start = v[0]
    communities = []
    to_visit = v
    visited = []
    queue = [start]
    while to_visit:
        check = queue[0]
        if check not in visited:
            visited.append(check)
            inter = visited + queue
            children = list(set(cgraph[check]).difference(inter))
            queue.extend(children)
            queue.pop(0)
            to_visit.remove(check)
            if len(queue) == 0 and len(to_visit) != 0:
                communities.append(visited)
                visited = []
                queue.append(to_visit[0])
            elif len(queue) == 0 and len(to_visit) == 0:
                communities.append(visited)

    list_community_graph = {}
    community_index = 1
    for community in communities:
        keys = community
        list_community_graph[community_index] = {x: cgraph[x] for x in keys}
        community_index += 1

    return list_community_graph


def modularity(comm_list, edg, orig_degree):
    m = len(edg)
    modularity_num = 0
    for comm in comm_list:
        degree_community = {}
        community = comm_list[comm]
        for node in community:
            degree_community[node] = len(community[node])
        edges_list = [(k, v) for k, v in community.items()]
        # print(edges_list)
        comm_edge = []
        for i in edges_list:
            for j in i[1]:
                comm_edge.append((i[0], j))
                comm_edge.append((j, i[0]))
        for i in community:
            for j in community:
                if (i, j) in edg:
                    a = 1
                else:
                    a = 0
                inner_value = (a - (orig_degree[i] * orig_degree[j]) / (2 * m))
                modularity_num += inner_value
    final_modularity = modularity_num / (2 * m)
    return final_modularity


threshold = int(sys.argv[1])
input_file = sys.argv[2]
betweenness_output_file = sys.argv[3]
community_output_file = sys.argv[4]

sc = SparkContext.getOrCreate(
    SparkConf().setMaster("local[*]").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g"))
sc.setLogLevel('WARN')

ub_sample = sc.textFile(input_file)
header = ub_sample.first()
ub_sample = ub_sample.filter(lambda x: x != header).map(lambda x: tuple(x.split(","))).distinct().persist()

filtered_uid_bid = ub_sample.combineByKey(to_list, append, extend).map(lambda x: (x[0], list(set(x[1])))).filter(
    lambda x: len(list(set(x[1]))) >= threshold).collectAsMap()

uid_pairs = list(combinations(sorted(filtered_uid_bid.keys()), 2))
edges = []
vertex_list = []

for i in uid_pairs:
    user1_list = filtered_uid_bid[i[0]]
    user2_list = filtered_uid_bid[i[1]]
    co_bid = list(set(user1_list) & set(user2_list))
    if len(co_bid) >= threshold:
        edges.append(i)
        edges.append((i[1], i[0]))
        vertex_list.extend([(i[0],), (i[1],)])

vertex_list = list(set(vertex_list))
graph_actual = create_graph(edges)

actual_degree = {}
for node in graph_actual:
    actual_degree[node] = len(graph_actual[node])

betweenness = girvan_newman(graph_actual)
with open(betweenness_output_file, 'w+') as fp:
    for i in betweenness:
        fp.writelines(str(i)[1:-1] + "\n")

current_graph = copy.deepcopy(graph_actual)
count = 0
iteration = {}
modularity_dict = {}
old_community_no = 1

while betweenness:
    count += 1
    edge_removed_graph = edge_removal(current_graph, betweenness)
    copy_edge_removed_graph = copy.deepcopy(edge_removed_graph)
    vertices = list(copy_edge_removed_graph.keys())
    single_communities = []
    for c in range(0, len(vertices)):
        if not copy_edge_removed_graph[vertices[c]]:
            single_communities.append(vertices[c])
            del copy_edge_removed_graph[vertices[c]]
    if len(list(copy_edge_removed_graph.keys())) == 0:
        break

    communities_created_list = check_connected(edge_removed_graph, list(edge_removed_graph.keys()))
    community_list_wo_singles = copy.deepcopy(communities_created_list)

    no_of_communities_generated = len(community_list_wo_singles)
    single_dict = {}
    index = no_of_communities_generated + 1
    for i in single_communities:
        single_dict[index] = {i: []}
        index += 1

    final_communities = {**community_list_wo_singles, **single_dict}
    iteration[count] = final_communities
    new_community_no = len(list(final_communities.keys()))
    betweenness = []
    if new_community_no > old_community_no:
        old_community_no = new_community_no
        modularity_dict[count] = modularity(communities_created_list, edges, actual_degree)

        for no in communities_created_list:
            if len(list(communities_created_list[no].values())[0]) == 0:
                continue
            else:
                betweenness.extend(girvan_newman(communities_created_list[no]))
        betweenness = sorted(betweenness, key=lambda x: (-x[1], x[0][0]))

    else:
        for no in communities_created_list:
            if len(list(communities_created_list[no].values())[0]) == 0:
                continue
            else:
                betweenness.extend(girvan_newman(communities_created_list[no]))
        betweenness = sorted(betweenness, key=lambda x: (-x[1], x[0][0]))

mod_list = [(k, v) for k, v in modularity_dict.items()]
mod_list = sorted(mod_list, key=lambda x: (-x[1]))
max_modularity_comm = iteration[mod_list[0][0]]
print("Max Modularity",mod_list[0])

with open("community_dict.txt", 'w+') as fp:
    json.dump(iteration, fp)


mod_comm = []
for i in max_modularity_comm:
    mod_comm.append(sorted(max_modularity_comm[i].keys()))

mod_result = sorted(mod_comm, key=lambda x: (len(x),x))


with open(community_output_file, 'w+') as fp:
    for i in mod_result:
        fp.writelines(str(i)[1:-1] + "\n")

fp.close()


print("Duration: ", time.time() - start)
