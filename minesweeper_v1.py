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
# random=True notifies that  the algo plays with probas
def clickOn(x, y, random=False):
    global edges, gameover, status
    value = field[x, y]

    # In case the algo plays with probas, it has a chance to fail
    if random and value == -1:
        if has_display:
            # Colors the cell in red
            tk.Label(root, text=' '+str(value)+' ', relief='sunken', bg='#ff0000',
                        borderwidth=2).grid(row=x, column=y)
        gameover = True

    else:
        if has_display:
            # Updates graphical view when clicking on non-mine cell
            tk.Label(root, text=' '+str(value)+' ' if value != 0 else '   ', relief='sunken', bg='#bfbfbf',
                    borderwidth=2).grid(row=x, column=y)

        status[x, y] = value
        # Adds the clicked cell to the edges list, or expanding the view if it is not an edge
        if field[x, y] != 0:
            edges.append([x, y])
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
        cond = lambda x, y: status[x, y] is not None
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
    global edges, status
    value = field[x, y]

    if has_display:
        # Updates graphical view
        tk.Label(root, text=' '+str(value)+' ' if value != 0 else '   ', relief='sunken', bg='#bfbfbf',
                borderwidth=2).grid(row=x, column=y)
            
    status[x, y] = value
    # Calling autoClickOn() recursively if the neighbors also have a "0" value
    if field[x, y] == 0:
        neighbors = getNeighbors(x, y)
        for neighbor in neighbors:
            x_neighbor, y_neighbor = neighbor
            if status[x_neighbor, y_neighbor] is None:
                autoClickOn(x_neighbor, y_neighbor)
    # Adding the neighbors with a non-"0" value to the edges list
    else:
        edges.append([x, y])



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
    
    # If it is not an edge anymore (it has not hidden neighbors anymore)
    if nb_hidden == 0:
        return 0
   
    return (status[x, y] - nb_nearby_flags) / nb_hidden



# Clicks on the cell with the smaller probability to have a "-1" value (mine)
def probabilisticClick():
    global edges
    if Verbose:
        print('GOING RANDOM')

    # Finds the edge with the minimal estimated proba to have a "-1" value nearby
    min_proba = 1
    min_edge = edges[0]
    for edge in edges:
        proba = computeProba(edge[0], edge[1])
        if proba < min_proba and proba > 0:
            min_proba = proba
            min_edge = edge
    min_hidden_neighbors = getNeighbors(min_edge[0], min_edge[1], state=0)

    # If there is a wall of mines, re-initialize edges so that a random call will be made in main()
    if(len(min_hidden_neighbors) == 0):
        print('blocked')
        print(nb_flags)
        print(nb_mines)
        print(count_blocked)
        edges = deque([])
    
    else:
        # For each hidden neighbor of this edge, finds the one which minimizes the maximum proba of 
        # having a "-1" value given its visible neighbors
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
        clickOn(x_min_hidden_neighbor, y_min_hidden_neighbor, random=True)



# Sweeps mines in a deterministic way (given the values of the edges, right-clicks on cells with a 1-probability of having a "-1" value, and clicks on cells with a 0-probability of having a non-"-1" value
def sweepMines(x, y):
    global count_blocked, edges
    hidden_neighbors = getNeighbors(x, y, state=0)
    nb_nearby_flags = len(getNeighbors(x, y, state=-1))

    # If we already discovered all the mines in the neighborhood of the cell
    if nb_nearby_flags == status[x, y]:
        # Notify that the algo is not blocked
        count_blocked = 0
        # Clicks on all the remaining hidden neighbors
        for hidden_neighbor in hidden_neighbors:
            clickOn(hidden_neighbor[0], hidden_neighbor[1])
    
    else:
        nb_mines_left = status[x, y] - nb_nearby_flags
        nb_hidden_neighbors = len(hidden_neighbors)
        
        # If there are as many mines left as hidden neighbors, right-click on all of them
        if nb_mines_left == nb_hidden_neighbors:
            # Notifies that the algo is not blocked
            count_blocked = 0
            for hidden_neighbor in hidden_neighbors:
                rightClickOn(hidden_neighbor[0], hidden_neighbor[1])
        
        # If nothing can be done so far
        else:
            # Adds the cell to the end of the list, and notifies that the algo did not move on
            edges.appendleft([x, y])
            count_blocked += 1



# Main loop
def main(start=False):
    global result, edges

    # If it is the first pass in the loop, choose the first cell among those with a "0" value
    if start:
        x = random.randint(0, rows-1)
        y = random.randint(0, cols-1)
        while field[x][y] != 0:
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
        probabilisticClick()

        if has_display:
            # Loop after counter ms    
            root.after(counter, main)
    
    else:
        if Verbose:
            print('WALL OF MINES???')
        
        x_hidden, y_hidden = 0, 0
        for x in range(rows):
            for y in range(cols):
                if status[x, y] is None:
                    x_hidden, y_hidden = x, y
                    break
                
        clickOn(x_hidden, y_hidden, random=True)
        if has_display:
            # Displays current cell in yellow
            changeColor(x_hidden, y_hidden)

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
