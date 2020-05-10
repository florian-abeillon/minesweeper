import argparse
import random
import time
from collections import deque

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

##########################################################################################################################################################################################################################

#################################################
########## PREPARE THE ENVIRONMENT ##############
#################################################


# Prepares the play grid
def prepareGame():
    global field, status
    # field is the play grid: -1 represents a bomb, and a positive number is the number of nearby mines
    # status is a copy of the play grid: None means that the cell has not been clicked on (it is "hidden"), -1 means that it has been tagged with a flag, and 1 means that we clicked on this cell ("visible")
    field, status = np.zeros((rows, cols), dtype=int), np.full((rows, cols), None)

    # Generates mines
    for _ in range(nb_mines):
        x = random.randint(0, rows-1)
        y = random.randint(0, cols-1)
        # Prevent mines spawning on top of each other
        while field[x, y] == -1:
            x = random.randint(0, rows-1)
            y = random.randint(0, cols-1)
        field[x, y] = -1

        # Modifies the neighbors' values accordingly
        if x != 0:
            if y != 0 and field[x-1, y-1] != -1:
                    field[x-1, y-1] = field[x-1, y-1] + 1
            if field[x-1, y] != -1:
                field[x-1, y] = field[x-1, y] + 1
            if y != cols-1 and field[x-1, y+1] != -1:
                    field[x-1, y+1] = field[x-1, y+1] + 1

        if y != 0 and field[x, y-1] != -1:
                field[x, y-1] = field[x, y-1] + 1

        if y != cols-1 and field[x, y+1] != -1:
                field[x, y+1] = field[x, y+1] + 1

        if x != rows-1:
            if y != 0 and field[x+1, y-1] != -1:
                    field[x+1, y-1] = field[x+1, y-1] + 1
            if field[x+1, y] != -1:
                field[x+1, y] = field[x+1, y] + 1
            if y != cols-1 and field[x+1, y+1] != -1:
                    field[x+1, y+1] = field[x+1, y+1] + 1



##########################################################################################################################################################################################################################

#################################################
############## USEFUL FUNCTIONS #################
#################################################


# Clicks on a hidden cell
def clickOn(x, y):
    global edges, gameover, status, edge_map
    value = field[x, y]

    # In case the algo plays with probas, it has a chance to fail
    if value == -1:
        if has_display:
            # Colors the cell in red
            tk.Label(root, text=' '+str(value)+' ', relief='sunken', bg='#ff0000',
                        borderwidth=2).grid(row=x, column=y)
        gameover = True

    else:
        edge_map[x, y, 0] = 1
        if has_display:
            # Updates graphical view when clicking on non-mine cell
            tk.Label(root, text=' '+str(value)+' ' if value != 0 else '   ', relief='sunken', bg='#bfbfbf',
                    borderwidth=2).grid(row=x, column=y)

        status[x, y] = value
        # Adds the clicked cell to the edges list, or expanding the view if it is not an edge
        if field[x, y] != 0:
            edges.append((x, y))
        else:
            autoClickOn(x, y)



# Returns all the neighbors of a cell
# state is None if we want every neighbor; state > 0 if we want only the visible ones; state = 0 if we want only the hidden ones; state < 0 if we want only the ones tagged with a flag
# scenario is status (the current game), or a copy of status depending on the hypothetical scenario the algo is computing
def getNeighbors(x, y, state=None, scenario=None):
    neighbors = []
    if scenario is not None:
        map_status = scenario
    else:
        map_status = status

    # Returns every neighbor
    if state is None:
        cond = lambda x, y: True
    # Returns every visible neighbor
    elif state > 0:
        cond = lambda x, y: map_status[x, y] is not None and map_status[x, y] >= 0
    # Returns every hidden neighbor
    elif state == 0:
        cond = lambda x, y: map_status[x, y] is None
    # Returns every neighbor tagged with a flag
    else:
        cond = lambda x, y: map_status[x, y] == -1

    if x != 0 and y != 0 and cond(x-1, y-1):
        neighbors.append((x-1, y-1))
    if x != 0 and cond(x-1, y):
        neighbors.append((x-1, y))
    if x != 0 and y != cols-1 and cond(x-1, y+1):
        neighbors.append((x-1, y+1))
    if y != 0 and cond(x, y-1):
        neighbors.append((x, y-1))
    if y != cols-1 and cond(x, y+1):
        neighbors.append((x, y+1))
    if x != rows-1 and y != 0 and cond(x+1, y-1):
        neighbors.append((x+1, y-1))
    if x != rows-1 and cond(x+1, y):
        neighbors.append((x+1, y))
    if x != rows-1 and y != cols-1 and cond(x+1, y+1):
        neighbors.append((x+1, y+1)) 

    return neighbors



# When clicking on a cell with a "0" value, expanding the view until the edge
def autoClickOn(x, y):
    global edges, status, edge_map
    value = field[x, y]

    if has_display:
        # Updates graphical view
        tk.Label(root, text=' '+str(value)+' ' if value != 0 else '   ', relief='sunken', bg='#bfbfbf',
                borderwidth=2).grid(row=x, column=y)
            
    status[x, y] = value
    # Calling autoClickOn() recursively if the neighbors also have a "0" value
    if field[x, y] == 0:
        edge_map[x, y, 0] = 0
        neighbors = getNeighbors(x, y)
        for neighbor in neighbors:
            x_neighbor, y_neighbor = neighbor
            if status[x_neighbor, y_neighbor] is None:
                autoClickOn(x_neighbor, y_neighbor)
    # Adding the neighbors with a non-"0" value to the edges list
    else:
        edge_map[x, y, 0] = 1
        edges.append((x, y))



# Adds a flag on supposed mines
def rightClickOn(x, y):
    global nb_flags, status

    if has_display:
        # Updates graphical view (colors the cell in blue)
        tk.Label(root, text=' X ', relief='raised', bg='#0400ff',
                borderwidth=2).grid(row=x, column=y)

    nb_flags += 1
    status[x, y] = -1



# Computes proba estimation: the number of nearby mines left / hidden neighbors
def computeProba(x, y):
    nb_hidden, nb_nearby_flags = len(getNeighbors(x, y, state=0)), len(getNeighbors(x, y, state=-1))

    return (status[x, y] - nb_nearby_flags) / nb_hidden



# Returns connected neighbors on the edge, and also their proba according to computeProba() if proba=True
def getConnectedEdges(connected_edges, x, y):
    global edge_map

    # Adds current edge to the list, and tags it as connected in edge_map
    connected_edges.append((x, y))
    # Compute the proba to have a "-1" value among the hidden neighbors
    edge_map[x, y, 1] = computeProba(x, y)
    edge_map[x, y, 0] = 2

    # Adds to the list all the connected edges of the neighbors also on the edge
    visible_neighbors = getNeighbors(x, y, state=1)
    for visible_neighbor in visible_neighbors:
        if edge_map[visible_neighbor][0] == 1:
            connected_edges += getConnectedEdges([], visible_neighbor[0], visible_neighbor[1])
    
    return connected_edges



# Returns connected neighbors on the edges that share common hidden edges, and also their proba according to computeProba() if proba=True
def getAlmostConnectedEdges(x, y):
    connected_edges = getConnectedEdges([], x, y)
    almost_connected_edges = set(connected_edges)
    hidden_edges = set([])
    is_already_in_set = False

    # We browse all hidden edges, trying to find other visible edges that are not in our current set
    while not is_already_in_set:
        is_already_in_set = True

        # Add the hidden frontier (union of one or several hidden edges sets)
        for almost_connected_edge in almost_connected_edges:
            hidden_neighbors = getNeighbors(almost_connected_edge[0], almost_connected_edge[1], state=0)
            for hidden_neighbor in hidden_neighbors:
                hidden_edges.add(hidden_neighbor)
        
        # Add the visible frontier (union of one or several visible edges sets)
        for hidden_edge in hidden_edges:
            visible_neighbors = getNeighbors(hidden_edge[0], hidden_edge[1], state=1)
            for visible_neighbor in visible_neighbors:
                # If a visible edge is not in the current set, we shall add it to the set, and add all their hidden edges
                if visible_neighbor not in almost_connected_edges:
                    is_already_in_set = False
                    almost_connected_edges.update(getConnectedEdges([], visible_neighbor[0], visible_neighbor[1]))

    almost_connected_edges = list(almost_connected_edges)
    nb_almost_connected_edges = len(almost_connected_edges)
    for i in range(nb_almost_connected_edges):
        almost_connected_edges[i] = [almost_connected_edges[i], 0]

    return almost_connected_edges



# Computes an estimation of the probability to have a mine nearby
def computeConnectedEdgesProba(connected_edges): 
    # If the current edge does not have any connected neighbor
    if len(connected_edges) == 1:
        return connected_edges

    # For every connected edge, compute the probas gradient
    nb_connected_edges = len(connected_edges)
    for i in range(nb_connected_edges):
        # Storing each connected edge, and its maximum gradient value
        connected_edges[i][1] = computeMaxDif(connected_edges[i][0])
    
    return connected_edges



# Computes the difference between neighbor connected edges probas
def computeMaxDif(edge):
    max_dif = 0
    edge_proba = edge_map[edge][1]
    visible_neighbors = getNeighbors(edge[0], edge[1], state=1)

    # For each neighbor that is on the edge, compute the gradient (in absolute value)
    for visible_neighbor in visible_neighbors:
        if edge_map[visible_neighbor][0] == 2:
            abs_dif = abs(edge_proba - edge_map[visible_neighbor][1])
            # Keep the highest in memory
            if abs_dif > max_dif:
                max_dif = abs_dif
        
    return max_dif



# Deeply investigates (beginning with edges with higher gradients), and clicks when the algo is certain
def deepInvestigation(connected_edges):
    global count_blocked
    
    # The algo will click on a new cell (either randomly or not), thus it might unblock the situation
    count_blocked = 0

    hidden_edges = set([])
    for connected_edge in connected_edges:
        hidden_neighbors = getNeighbors(connected_edge[0][0], connected_edge[0][1], state=0)
        for hidden_neighbor in hidden_neighbors:
            hidden_edges.add(hidden_neighbor)
    hidden_edges = list(hidden_edges)
    nb_hidden_edges = len(hidden_edges)
    for i in range(nb_hidden_edges):
        hidden_edges[i] = [hidden_edges[i], 2]

    # For each connected edge, create all the possible scenarii and compute the estimation of the latent probabilities of their hidden neighbors
    for connected_edge in connected_edges:
        hidden_neighbors = computeLatentProbas(connected_edge[0])

        # For each hidden neighbor of each connected edge 
        for hidden_neighbor in hidden_neighbors:
            for hidden_edge in hidden_edges:
                if hidden_neighbor[0] == hidden_edge[0]:
                    if (hidden_edge[1] != 1 and hidden_neighbor[1] < hidden_edge[1]) or hidden_neighbor[1] == 1:
                        hidden_edge[1] = hidden_neighbor[1]
                    break

    has_moved_on = False
    for hidden_edge in hidden_edges:
        if hidden_edge[1] == 0:
            clickOn(hidden_edge[0][0], hidden_edge[0][1])
            has_moved_on = True
            
        elif hidden_edge[1] == 1:
            rightClickOn(hidden_edge[0][0], hidden_edge[0][1])
            has_moved_on = True



    if not has_moved_on:
        has_moved_on_deep = False
        edges_with_best_proba = []
        connected_edges_deep = [connected_edge[0] for connected_edge in connected_edges]

        # Keep in memory the edges we checked in bigDeepClick
        connected_edges_checked = set(connected_edges_deep)
        deque_edges = edges.copy()

        # While bigDeepClick has not unblocked the situation, and there are still new edges to check, bigDeepClick() on the new edge
        while not has_moved_on_deep and deque_edges:
            has_moved_on_deep, edge_with_best_proba = bigDeepClick(connected_edges_deep)
            edges_with_best_proba.append(edge_with_best_proba)

            # If bigDeepClick() has still not unblocked the situation
            if not has_moved_on_deep:
                # While there are still edges to check, check if another frontier (to feed to bigDeepClick()) can be found
                while deque_edges:
                    edge = deque_edges.pop()
                    if not edge in connected_edges_checked:
                        connected_edges_deep = getConnectedEdges([], edge[0], edge[1])
                        break
                # If a new frontier has been found, add its cells in the set
                for connected_edge in connected_edges_deep:
                    connected_edges_checked.add(connected_edge)

        # If no frontier has allowed bigDeepClick() to unblock the situation, make a probabilistic guess on the cell which has the best probability
        if not has_moved_on_deep:
            probabilisticClick(edges_with_best_proba)




# Computes proba estimation of edge
def computeLatentProbas(edge):
    # Creates a list of possible scenarii (this list is actually a list of coordinates, representing mines position in a scenario)
    list_scenarii = createScenarii(edge)
    hidden_neighbors_list = getNeighbors(edge[0], edge[1], state=0)
    hidden_neighbors = np.array([[neighbor, 0] for neighbor in hidden_neighbors_list])
    nb_valid_cases = 0

    # For each scenario, check if it is possible, given the status of visible neighbors
    for scenario in list_scenarii:
        if testScenario(scenario, edge):
            nb_valid_cases += 1
            
            # If it is possible, count the number of times a mine was on a particular cell
            nb_hidden = len(hidden_neighbors)
            for i in range(nb_hidden):
                if hidden_neighbors[i, 0] in scenario:
                    hidden_neighbors[i, 1] = hidden_neighbors[i, 1] + 1
    
    if nb_valid_cases == 0:
        print(edge)
        print(hidden_neighbors)

    # After checking all possible scenarii, divide the counter by the number of valid scenarii to have a proba
    for hidden_neighbor in hidden_neighbors:
        hidden_neighbor[1] = hidden_neighbor[1] / nb_valid_cases

    return hidden_neighbors



# Creates scenarii
def createScenarii(edge):
    # Computing the number of mines left nearby
    nb_mines_left = status[edge] - len(getNeighbors(edge[0], edge[1], state=-1))
    hidden_neighbors = getNeighbors(edge[0], edge[1], state=0)

    # Creating all scenarii possible, given the number of mines left nearby, and the hidden neighbors
    if nb_mines_left == -1:
        print('nb_mines_left == -1 in createScenarii')
    return recursiveScenarii(hidden_neighbors, nb_mines_left)



# Creates scenarii recursively
def recursiveScenarii(hidden_neighbors, nb_mines_left):
    if nb_mines_left == -1:
        print("nb_mines_left == -1")
        for x in range(rows):
            for y in range(cols):
                if status[x, y] is None:
                    print([x, y])

        # import tkinter as tk
        # root = tk.Tk()
        # for x in range(rows):
        #     for y in range(cols):
        #         value = field[x, y]
        #         if status[x, y] is not None:
        #             tk.Label(root, text=' '+str(value)+' ' if value != 0 else '   ', relief='sunken', bg='#bfbfbf',
        #                 borderwidth=2).grid(row=x, column=y)
        #         else:
        #             tk.Label(root, text='   ', relief='raised', bg='#9f9f9f',
        #                 borderwidth=2).grid(row=x, column=y)
        for hidden_neighbor in hidden_neighbors:
            value = field[hidden_neighbor[0], hidden_neighbor[1]]
            tk.Label(root, text=' '+str(value)+' ' if value != 0 else '   ', relief='sunken', bg='#fbff00',
                borderwidth=2).grid(row=x, column=y)
    # If no mines left nearby, return an empty list (as there are no mines)
    if nb_mines_left == 0:
        return [[]]
    
    # If there are as many mines left nearby as hidden neighbors, return the hidden neighbors
    elif len(hidden_neighbors) == nb_mines_left:
        return [hidden_neighbors]

    scenarii = []
    # Beginning to create scenarii by supposing that the first hidden neighbor is (or is not) a mine, and calling the function recursively to finish to build the scenario
    iter_max = len(hidden_neighbors) - nb_mines_left + 1
    for i in range(iter_max):
        if nb_mines_left == 0:
            print('nb_mines_left == -1 in recursiveScenarii')
        list_subcases = recursiveScenarii(hidden_neighbors[i+1:], nb_mines_left - 1)

        # For each subcase returned by the recursive call, add the first hidden neighbor considered in the loop
        for subcase in list_subcases:
            subcase.append(hidden_neighbors[i])
            # Add the scenario to the list of scenarii
            scenarii.append(subcase)
    
    return scenarii



# Test validity of scenario
def testScenario(scenario, edge):
    visible_neighbors_to_check = set([])

    # Create a set of all visible neighbors of all hidden neighbors of current edge
    hidden_neighbors = getNeighbors(edge[0], edge[1], state=0)
    for hidden_neighbor in hidden_neighbors:
        visible_neighbors = getNeighbors(hidden_neighbor[0], hidden_neighbor[1], state=1)
        for visible_neighbor in visible_neighbors:
            visible_neighbors_to_check.add(visible_neighbor)

    # For all visible neighbor in the set, check how many mines are left nearby
    for visible_neighbor_to_check in visible_neighbors_to_check:
        nb_mines_left = status[visible_neighbor_to_check[0], visible_neighbor_to_check[1]] - len(getNeighbors(visible_neighbor_to_check[0], visible_neighbor_to_check[1], state=-1))
        hidden_neighbors_to_check = getNeighbors(visible_neighbor_to_check[0], visible_neighbor_to_check[1], state=0)
        nb_hidden_neighbors_to_check = len(hidden_neighbors_to_check)

        # For each hidden neighbor of the visible neighbors in the set, count the number of mines and hidden neighbors left if the scenario were True
        for hidden_neighbor_to_check in hidden_neighbors_to_check:
            if hidden_neighbor_to_check in hidden_neighbors:
                nb_hidden_neighbors_to_check -= 1
                if hidden_neighbor_to_check in scenario:
                    nb_mines_left -= 1

        # If the scenario forces the algo to put too many flags, or if there are not enough hidden neighbors given the number of mines left, the scenario is not possible
        if nb_mines_left < 0 or nb_hidden_neighbors_to_check < nb_mines_left:
            return False

    # If the scenario passes the test on every visible neighbor in the set, it is considered as possible
    return True



# Tries to find cells with a 1-probability of being/not being mines; if it does not find any, clicks on the cell with the smallest estimated probability of being a mine
def deepClick():
    global edges 
    if Verbose:
        print('WE NEED TO GO DEEPER')
        print('~GOING DEEPER~')
        print('')

    # Takes the first edge that is connected with other edges
    x, y = edges.pop()
    nb_hidden_neighbors = len(getNeighbors(x, y, state=0))
    # Not an edge anymore
    while nb_hidden_neighbors == 0:
        x, y = edges.pop()
        nb_hidden_neighbors = len(getNeighbors(x, y, state=0))

    # Puts it back in the queue
    edges.append((x, y))

    # Gets the list of connected edges, and the proba to have mines among their hidden neighbors
    connected_edges = getAlmostConnectedEdges(x, y)
    connected_edges_with_gradients = computeConnectedEdgesProba(connected_edges)

    # Sorts them according to their proba gradients
    connected_edges_with_gradients.sort(key=lambda el: el[1], reverse=True)

    # Tries to find (non) "-1" values with probability 1; if there is none, clicks on the cell with the lowest probability to have a "-1" value
    deepInvestigation(connected_edges_with_gradients)

    # Resets connections between edges
    for x in range(rows):
        for y in range(cols):
            if edge_map[x, y][0] == 2:
                edge_map[x, y][0] = 1



# Determines all valid cases with hypotheses on the position of mines among hidden edges
def determineCases(remaining_edges, map_scenario, mines_scenario):
    global mines_valid_scenarii

    # If we have checked all edges and the case passed all tests, adds it to the list of valid cases
    if len(remaining_edges) == 0:
        if len(mines_scenario) <= nb_mines - nb_flags:
            mines_valid_scenarii.append(mines_scenario)
        return

    remaining_edges_case = remaining_edges.copy()
    # Take an edge, determine its hidden neighbors and its actualized status in this scenario
    edge = remaining_edges_case.pop()
    hidden_neighbors_scenario = getNeighbors(edge[0], edge[1], state=0, scenario=map_scenario)
    status_scenario = status[edge] - len(getNeighbors(edge[0], edge[1], state=-1, scenario=map_scenario))

    # If there are more mines nearby than hidden neighbors, or if there are too many flags around it (given its initial status), the scenario is not valid
    if status_scenario < 0 or status_scenario > len(hidden_neighbors_scenario):
        return
    
    # Create all possible combinations of mines positions nearby
    if status_scenario == -1:
        print('nb_mines_left == -1 in determineCases')
    edge_scenarii = recursiveScenarii(hidden_neighbors_scenario, status_scenario)

    # For each combination, create the according scenario by actualizing its status map and list of mines
    for edge_scenario in edge_scenarii:
        new_remaining_edges = remaining_edges_case.copy()
        map_new_scenario = map_scenario.copy()
        mines_new_scenario = mines_scenario.copy()
        mines_new_scenario += edge_scenario

        # Actualizing "-1" values positions in status map
        for mine in edge_scenario:
            map_new_scenario[mine] = -1

        # Actualizing non "-1" values positions in status map
        for hidden_neighbor_scenario in hidden_neighbors_scenario:
            if map_new_scenario[hidden_neighbor_scenario] is None:
                map_new_scenario[hidden_neighbor_scenario] = 0
        
        # Continue the case until there is no remaining edge left or the scenario is not valid
        determineCases(new_remaining_edges, map_new_scenario, mines_new_scenario)
    
    return



# Computes all possible scenarii of mines positions on the edge, and either finds a deterministic solution, or clicks on the cell determined with the highest proba
def bigDeepClick(connected_edges):
    global mines_valid_scenarii
    if Verbose:
        print("YOU MUSTN'T BE AFRAID TO CLICK A LITTLE DEEPER DARLING")
        print('~CLICKING EVEN DEEPER~')
        print('')
    
    # Gets the list of hidden edges
    hidden_edges = set([])
    
    nb_connected_edges = len(connected_edges)
    for i in range(nb_connected_edges):        
        if has_display:
            tk.Label(root, text=' '+str(field[connected_edges[i][0], connected_edges[i][1]])+' ', relief='sunken', bg='#ff00f2',
                borderwidth=2).grid(row=connected_edges[i][0], column=connected_edges[i][1])

        hidden_neighbors = getNeighbors(connected_edges[i][0], connected_edges[i][1], state=0)
        for hidden_neighbor in hidden_neighbors:
            hidden_edges.add(hidden_neighbor)
    hidden_edges = list(hidden_edges)

    if len(hidden_edges) == 0:
        print(connected_edges)
        print(status)

    # Creates a dict with the probas of each hidden edge
    probas = dict(zip(hidden_edges, [0] * len(hidden_edges)))


    # Computes all possible scenarii, and stores them in mines_valid_scenarii
    map_scenario = status.copy()
    mines_valid_scenarii = []
    determineCases(connected_edges, map_scenario, [])

    # For each scenario, increments the counter of each cell having a "-1" value in this scenario
    for mines_valid_scenario in mines_valid_scenarii:
        for mine in mines_valid_scenario:
            probas[mine] += 1


    nb_valid_scenarii = len(mines_valid_scenarii)
    proba = 0
    max_hidden_edge = hidden_edges[0]
    has_moved_on = False

    # For each hidden edge, divides the "proba" by the number of scenarii, so as to have a proba
    for hidden_edge in hidden_edges:
        if nb_valid_scenarii == 0:
            print('connected_edges', connected_edges)
            print('hidden_edges', hidden_edges)
            break
        probas[hidden_edge] = probas[hidden_edge] / nb_valid_scenarii

        # If the proba is 0, clicks on it
        if probas[hidden_edge] == 0:
            has_moved_on = True
            clickOn(hidden_edge[0], hidden_edge[1])

        # If the proba is 1, right-clicks on it
        elif probas[hidden_edge] == 1:
            has_moved_on = True
            rightClickOn(hidden_edge[0], hidden_edge[1])

        # If the proba is closer to 0 or 1 than proba, store this edge as the one which minimizes the uncertainty
        elif abs(probas[hidden_edge] - 0.5) > abs(proba):
            proba = probas[hidden_edge] - 0.5
            max_hidden_edge = hidden_edge
    
    # If the scenarii couldn't unblock the situation, (right-)clicks on the edge which minimizes the uncertainty
    if not has_moved_on:
        return False, (max_hidden_edge, proba)
    else:
        return True, None




# Finds the cell with the best probability whithin a list, and (right-)clicks on it
def probabilisticClick(edges_with_best_proba):
    if Verbose:
        print("ARE YOU WILLING TO TAKE A LEAP OF FAITH, OR BECOME AN OLD MAN FILLED WITH REGRET, WAITING TO DIE ALONE?")
        print("~TAKING A LEAP OF FAITH BECAUSE IT DOESN'T WANT TO BECOME AN OLD MAN FILLED WITH REGRET, WAITING TO DIE ALONE~")
        print('')

    # Find the edge with the closest probability  to 0/1
    edge_with_best_proba, best_proba = edges_with_best_proba[0]
    for edge in edges_with_best_proba:     
        if abs(edge[1] - 0.5) > abs(best_proba - 0.5):
            edge_with_best_proba, best_proba = edge

    # Depending on the proba being close to 0 or 1, click or right-click
    if best_proba > 0:
        rightClickOn(edge_with_best_proba[0], edge_with_best_proba[1])
    else:
        clickOn(edge_with_best_proba[0], edge_with_best_proba[1])




# Sweeps mines in a deterministic way (given the values of the edges, right-clicks on cells with a 1-probability of having a "-1" value, and clicks on cells with a 0-probability of having a non-"-1" value
def sweepMines(x, y):
    global count_blocked, edges, edge_map
    hidden_neighbors = getNeighbors(x, y, state=0)
    nb_nearby_flags = len(getNeighbors(x, y, state=-1))

    # If we already discovered all the mines in the neighborhood of the cell
    if nb_nearby_flags == status[x, y]:
        # Notifies that the algo is not blocked
        count_blocked = 0
        edge_map[x, y, 0] = 0
        # Clicks on all the remaining hidden neighbors
        for hidden_neighbor in hidden_neighbors:
            clickOn(hidden_neighbor[0], hidden_neighbor[1])
    
    else:
        nb_mines_left = status[x, y] - nb_nearby_flags
        nb_hidden_neighbors = len(hidden_neighbors)
        
        # If there are as many mines left as hidden neighbors, right-click on all of them
        if nb_mines_left == nb_hidden_neighbors:
            count_blocked = 0
            edge_map[x, y, 0] = 0
            # Notifies that the algo is not blocked
            for hidden_neighbor in hidden_neighbors:
                rightClickOn(hidden_neighbor[0], hidden_neighbor[1])
        
        # If nothing can be done so far
        else:
            # Adds the cell to the end of the list, and notifies that the algo did not move on
            edges.appendleft((x, y))
            count_blocked += 1



# Main loop
def main(start=False):
    global result, edges

    # If it is the first pass in the loop, choose the first cell among those with a "0" value
    if start:
        x = random.randint(0, rows-1)
        y = random.randint(0, cols-1)
        while field[x, y] != 0:
            x = random.randint(0, rows-1)
            y = random.randint(0, cols-1)
            
        clickOn(x, y)

        if has_display:
            # Displays current cell in yellow
            changeColor(x, y)

            # Loop after counter ms    
            root.after(counter, main)

    # If the algo failed
    if gameover:
        result = -1
        if Verbose:
            print('FAIL!')
        if has_display:
            root.destroy()
        return

    # If the algo tagged as many cells as there are mines in the grid, click on all the remaining cells
    elif nb_flags == nb_mines:
        for x in range(rows-1):
            for y in range(cols-1):
                if status[x, y] is None:
                    clickOn(x, y)

        # If the algo failed after all
        if gameover:
            result = -1
            if Verbose:
                print('FAIL!')
            if has_display:
                root.destroy()
            return
        # If the algo succeeded in its mission
        else:
            result = 1
            if Verbose:
                print('SUCCESS!')

            if has_display:
                root.destroy()
            return

    # If the algo is not blocked (it has not gone through all the edges list without doing anything), tries to sweep mines starting from the first edge on the list
    elif count_blocked < len(edges):
        edge = edges.pop()
        if has_display: 
            # Displays current cell in yellow
            changeColor(edge[0], edge[1])

        sweepMines(edge[0], edge[1])
        
        if has_display:
            # Loop after counter ms    
            root.after(counter, main)
    
    # If the algo is indeed blocked, and there are still edges left, tries to click on the cell with the smaller probability to have a "-1" value
    elif len(edges) > 0:
        deepClick()

        if has_display:
            # Loop after counter ms    
            root.after(counter, main)
    
    else:
        if Verbose:
            print('WALL OF MINES???')
        
        x = random.randint(0, rows-1)
        y = random.randint(0, cols-1)
        while status[x, y] is not None:
            x = random.randint(0, rows-1)
            y = random.randint(0, cols-1)
            
        clickOn(x, y)
        if has_display:
            # Displays current cell in yellow
            changeColor(x, y)

            # Loop after counter ms    
            root.after(counter, main)



# Changes the color of the cell in yellow for counter ms
def changeColor(x, y):
    value = field[x, y]
    
    tk.Label(root, text=' '+str(value)+' ' if value != 0 else '   ', relief='sunken', bg='#fbff00',
            borderwidth=2).grid(row=x, column=y)

    root.after(counter, lambda : tk.Label(root, text=' '+str(value)+' ' if value != 0 else '   ', relief='sunken', bg='#bfbfbf',
            borderwidth=2).grid(row=x, column=y))






##########################################################################################################################################################################################################################

#################################################
################## LAUNCHER #####################
#################################################

    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--grid", help="""Type of grid: is it the "stanford" one, a 16*16 grid with 40 mines? Or is it the "expert" one, a 16*32 grid with 99 mines?""")
    parser.add_argument("-n", "--number", help="""Number of epochs""")
    parser.add_argument("-d", "--display", help="""Whether you want to display the grid""", action="store_true")
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_arguments()

    rows, cols, nb_mines = (16, 32, 99) if args.grid == 'expert' else (16, 16, 40)
    if args.display:
        has_display, Verbose = True, True
        counter = 100
        nb_epochs = 1
    else:
        has_display, Verbose = False, False
        nb_epochs = 10000
    
    if args.number:
        nb_epochs = args.number

    avg_success, avg_run_time, nb_success = 0, 0, 0
    x, y_avg_success, y_avg_run_time = [], [], []
    nb_errors = 0

    for epoch in range(1, nb_epochs+1):
        try:
            # Initializes values
            edges = deque([])
            edge_map = np.zeros((rows, cols, 2))
            mines_valid_scenarii = []
            nb_flags = 0
            count_blocked = 0
            gameover = False
            result = 0

            # Prepares new play grid, and graphical view
            prepareGame()

            if has_display:
                import tkinter as tk

                root = tk.Tk()
                for x in range(rows):
                    for y in range(cols):
                            tk.Label(root, text='   ', relief='raised', bg='#9f9f9f',
                                borderwidth=2).grid(row=x, column=y)

            # Running the loop while computing the duration
            start = time.time()
            if has_display:
                main(start=True)
                root.mainloop()
            else:
                main(start=True)
                while result == 0:
                    main()

            result = (result + 1) / 2
            duration = time.time() - start
            avg_success += result
            if result == 1:
                nb_success += 1
                avg_run_time += duration

            if epoch % 100 == 0:  # (nb_epochs / 100) == 0:
                x.append(epoch)
                y_avg_success.append(100 * avg_success / (epoch - nb_errors))
                y_avg_run_time.append(1000 * avg_run_time / nb_success)

                print('Epoch ', epoch, ':')
                print('Success rate:', 100 * avg_success / (epoch - nb_errors), '%')
                print('Average running time:', 1000 * avg_run_time / nb_success, 'ms')
                print('')

        except:
            nb_errors += 1
            pass


    if nb_epochs > 1:
        # Plotting average success rate and running time
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Average success rate", "Average running time")
        )
        fig.add_trace(
            go.Scatter(x=x, y=y_avg_success),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=y_avg_run_time),
            row=1, col=2
        )
        fig.update_xaxes(title_text="Number of tries", row=1, col=1)
        fig.update_yaxes(title_text="Total average success rate (in %)", range=[0, 100], row=1, col=1)
        fig.update_xaxes(title_text="Number of tries", row=1, col=2)
        fig.update_yaxes(title_text="Total average running time (in ms)", row=1, col=2)

        fig.update_layout(height=600, width=800, title_text='Average success with a '+str(rows)+'x'+str(cols)+' grid, and '+str(nb_mines)+' mines')
        fig.show()
