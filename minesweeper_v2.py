import argparse
import random
import time
from collections import deque

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
def getNeighbors(x, y, state=None):
    neighbors = []

    # Returns every neighbor
    if state is None:
        cond = lambda x, y: True
    # Returns every visible neighbor
    elif state > 0:
        cond = lambda x, y: status[x, y] is not None and status[x, y] >= 0
    # Returns every hidden neighbor
    elif state == 0:
        cond = lambda x, y: status[x, y] is None
    # Returns every neighbor tagged with a flag
    else:
        cond = lambda x, y: status[x, y] == -1

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
    proba = (status[x, y] - nb_nearby_flags) / nb_hidden
    
    return proba



# Returns the connected neighbors on the edge
def getConnectedEdges(connected_edges, x, y):
    global edge_map

    # Adds current edge to the list, and tags it as connected in edge_map
    connected_edges.append([(x, y), 0])
    edge_map[x, y, 0] = 2
    # Compute the proba to have a "-1" value among the hidden neighbors
    edge_map[x, y, 1] = computeProba(x, y)

    # Adds to the list all the connected edges of the neighbors also on the edge
    visible_neighbors = getNeighbors(x, y, state=1)
    for visible_neighbor in visible_neighbors:
        if edge_map[visible_neighbor][0] == 1:
            connected_edges += getConnectedEdges([], visible_neighbor[0], visible_neighbor[1])
    
    return connected_edges



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

    # For each connected edge, create all the possible scenarii and compute the estimation of the latent probabilities of their hidden neighbors
    for connected_edge in connected_edges:
        hidden_neighbors = computeLatentProbas(connected_edge[0])
        has_moved_on = False

        # For each hidden neighbor of each connected edge 
        for hidden_neighbor in hidden_neighbors:
            # If a hidden neighbor has a "-1" value with a probability 1
            if hidden_neighbor[1] == 1:
                rightClickOn(hidden_neighbor[0][0], hidden_neighbor[0][1])
                has_moved_on = True
            
            # If a hidden neighbor has a non-"-1" value with a probability 1
            elif hidden_neighbor[1] == 0:
                clickOn(hidden_neighbor[0][0], hidden_neighbor[0][1])
                has_moved_on = True

            # elif hidden_neighbor[1] < proba_edge_min:
            #     proba_edge_min = hidden_neighbor[1]

        # If the algo clicked on a cell (or several) in this loop, break the cell and stick with the first strategy
        if has_moved_on:
            return

    # If the algo cannot be certain of the value of at least one hidden case, it makes a probabilistic call
    probabilisticClick()



# Resets edge_map
def resetEdgeMap(connected_edges):
    global edge_map

    # Reset back indicator to 1 (edges are still edges, but we don't know if they are still connected)
    for connected_edge in connected_edges:
        edge_map[connected_edge[0]] = 1



# Computes proba estimation of edge
def computeLatentProbas(edge):
    # Creates a list of possible scenarii (this list is actually a list of coordinates, representing mines position in a scenario)
    list_scenarii = createScenarii(edge)
    hidden_neighbors = np.array([[neighbor, 0] for neighbor in getNeighbors(edge[0], edge[1], state=0)])
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
    return recursiveScenarii(hidden_neighbors, nb_mines_left)



# Creates scenarii recursively
def recursiveScenarii(hidden_neighbors, nb_mines_left):
    # If no mines left nearby, return an empty list (as there are no mines)
    if nb_mines_left == 0:
        return [[]]
    
    # If there are as many mines left nearby as hidden neighbors, return the hidden neighbors
    elif len(hidden_neighbors) == nb_mines_left:
        return [hidden_neighbors]

    else:
        scenarii = []
        # Beginning to create scenarii by supposing that the first hidden neighbor is (or is not) a mine, and calling the function recursively to finish to build the scenario
        iter_max = len(hidden_neighbors) - nb_mines_left + 1
        for i in range(iter_max):
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
    while nb_hidden_neighbors == 0:
        x, y = edges.pop()
        nb_hidden_neighbors = len(getNeighbors(x, y, state=0))

    # Puts it back in the queue
    edges.append((x, y))

    # Gets the list of connected edges, and the proba to have mines among their hidden neighbors
    connected_edges = getConnectedEdges([], x, y)
    connected_edges_with_gradients = computeConnectedEdgesProba(connected_edges)

    # Sorts them according to their proba gradients
    connected_edges_with_gradients.sort(key=lambda el: el[1], reverse=True)

    # Tries to find (non) "-1" values with probability 1; if there is none, clicks on the cell 
    # with the lowest probability to have a "-1" value
    deepInvestigation(connected_edges_with_gradients)

    # Resets connections between edges
    resetEdgeMap(connected_edges)



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



# Clicks on the cell with the smaller probability to have a "-1" value (mine)
def probabilisticClick():
    if Verbose:
        print("GOING RANDOM")

    # Finds the edge with the minimal estimated proba to have a "-1" value nearby
    min_proba = 1
    min_edge = edges[0]
    for edge in edges:
        proba = computeProba(edge[0], edge[1])
        if proba < min_proba:
            min_proba = proba
            min_edge = edge
    min_hidden_neighbors = getNeighbors(min_edge[0], min_edge[1], state=0)
    
    # For each hidden neighbor of this edge, finds the one which minimizes the maximum proba of having a "-1" value given its visible neighbors
    x_min_hidden_neighbor, y_min_hidden_neighbor = min_hidden_neighbors[0]
    min_max_proba = 1
    for hidden_neighbor in min_hidden_neighbors:
        max_proba = 0
        visible_neighbors = getNeighbors(hidden_neighbor[0], hidden_neighbor[1], state=1)

        # Finds the maximal proba of the cell having a "-1" value, given its visible neighbors
        for visible_neighbor in visible_neighbors:
            proba = computeProba(visible_neighbor[0], visible_neighbor[1])
            max_proba = proba if proba > max_proba else max_proba
        
        if max_proba < min_max_proba:
            min_max_proba = max_proba
            x_min_hidden_neighbor, y_min_hidden_neighbor = hidden_neighbor

    if Verbose:
        print("Minimal proba:", min_max_proba)
    clickOn(x_min_hidden_neighbor, y_min_hidden_neighbor)



# Main loop
def main(start=False):
    global edges, result

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



#######################################################


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

    for epoch in range(1, nb_epochs+1):
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
            y_avg_success.append(100 * avg_success / epoch)
            y_avg_run_time.append(1000 * avg_run_time / nb_success)

            print('Epoch ', epoch, ':')
            print('Success rate:', 100 * avg_success / epoch, '%')
            print('Average running time:', 1000 * avg_run_time / nb_success, 'ms')
            print('')


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
