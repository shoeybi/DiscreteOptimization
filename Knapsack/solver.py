#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

def calc_optimistic_value(capacity_,depth,value_sofar,values,weights):
    opt_value = float(value_sofar)
    capacity = capacity_
    for i in range(depth,len(values)):
        if capacity >= weights[i]:
            opt_value += float(values[i])
            capacity  -= weights[i]
        else:
            opt_value += float(values[i])/float(weights[i])*float(capacity)
            break
    return opt_value
    

def rec_solve(capacity_left,depth,value_sofar,values,weights,
              best_value_sofar,index_sofar,best_index,min_weights):

    assert capacity_left>=0
    # if we don't have any nodes left or the capacity_left is smaller than the minimum of weights, return
    if (depth == len(values)) or (capacity_left<min_weights):
        if value_sofar > best_value_sofar[0]:
             best_value_sofar[0] = value_sofar
             for i in range(depth):
                 best_index[i] = index_sofar[i]
             for i in range(depth,len(values),1):
                 best_index[i] = 0
        return

    # if we select the item
    if capacity_left >= weights[depth]:
        opt_value_select = calc_optimistic_value(capacity_left,depth,value_sofar,
                                                 values,weights)
    else:
       opt_value_select = -1.0;
    # and if we not
    opt_value_noselect = calc_optimistic_value(capacity_left,depth+1,value_sofar,
                                               values,weights)


    if opt_value_noselect>opt_value_select:    

        if opt_value_noselect>=best_value_sofar[0]:
            index_sofar[depth] = 0
            rec_solve(capacity_left,depth+1,value_sofar,values,weights,best_value_sofar,
                      index_sofar,best_index,min_weights)  

        if opt_value_select>=best_value_sofar[0]:
            index_sofar[depth] = 1
            rec_solve(capacity_left-weights[depth],depth+1,value_sofar+values[depth],
                      values,weights,best_value_sofar,index_sofar,best_index,min_weights)  
   
    else:
        
        if opt_value_select>=best_value_sofar[0]:
            index_sofar[depth] = 1
            rec_solve(capacity_left-weights[depth],depth+1,value_sofar+values[depth],
                      values,weights,best_value_sofar,index_sofar,best_index,min_weights)  
            
        if opt_value_noselect>=best_value_sofar[0]:
            index_sofar[depth] = 0
            rec_solve(capacity_left,depth+1,value_sofar,values,weights,best_value_sofar,
                      index_sofar,best_index,min_weights)  
    
    


def bb_solveIt(values,weights,capacity):
    # check sizes are compatible
    assert len(values) == len(weights)
    size = len(values)


    # print some info
    print ' > number of items:   ', len(values)
    print ' > capacity:          ', capacity
    print ' > minimum weight:    ', min(weights)
    print ' > maximum weight:    ', max(weights)

    # initialize the output
    value = 0
    taken = [0]*size

    # calcualte unit value of each item
    unit_values = [0.0]*size
    for i in range(size):
        unit_values[i] = float(values[i])/float(weights[i])

    # sort everything based on their unit values and keep the index
    index = [0]*size
    for i in range(size):
        index[i] = i
    # sort
    index_sorted   = [x for (y,x) in sorted(zip(unit_values,index))]
    values_sorted  = [0]*size
    weights_sorted = [0]*size
    for i in range(size):
        values_sorted[i]  = values[index_sorted[size-i-1]]
        weights_sorted[i] = weights[index_sorted[size-i-1]]

    

    index_sofar = [0]*size
    best_index  = [0]*size

    min_weights = min(weights)

    best_value_sofar = [0.0]*1;
    rec_solve(capacity,0,0.0,values_sorted,weights_sorted,best_value_sofar,index_sofar,best_index,min_weights)
    value = best_value_sofar[0]
    #print best_index
    for i in range(size):
        taken[index_sorted[i]] = best_index[size-i-1]
    return {"taken":taken,"value":int(value)}




def dp_solveIt(values,weights,capacity):
   
    # check sizes are compatible
    assert len(values) == len(weights)

    # initialize the output
    value = 0
    taken = []
    for i in range(len(values)):
        taken.append(0)
  
    # minimum available weight
    w_min = min(weights)
    
    # if min weight is larger than the capacity then no item can be filled in the knapsak
    if w_min > capacity:
        return

    # array dimension
    # increase dimension by one for padding by zero
    m = capacity - w_min + 2
    n = len(weights) + 1

    # check that we will not exceed a maximum memory in GB
    max_ram = 4
    memory  = (m*n*4.0)/(1024.0*1024.0) 

    # print some info
    print ' > number of items:   ', len(values)
    print ' > capacity:          ', capacity
    print ' > minimum weight:    ', w_min
    print ' > maximum weight:    ', max(weights)
    print ' > memory needed(MB): ', memory

    # check we will not exceed maximum memory specified
    if (m*n*1.0)>(max_ram*1024.0*1024.0*1024.0/4.0):
        raise ValueError('matrix is too big :-(')

    # create an array of zeros
    matrix = np.zeros((m,n),dtype='int32')

    # DP
    for j in range(1,n):
        wj = weights[j-1]
        for i in range(1,m):
            wt = w_min + i - 1
            if wj>wt:
                matrix[i,j] = matrix[i,j-1]
            else:
                ii = wt - wj - w_min + 1
                if ii<0:
                    ii = 0
                tmpvalue = values[j-1] + matrix[ii,j-1]
                if ( tmpvalue > matrix[i,j-1] ):
                    matrix[i,j] = tmpvalue
                else:
                    matrix[i,j] =  matrix[i,j-1]
                    
    # back-track to get the data
    i = m-1
    j = n-1
    while True:
        if (i<=0) or (j==0):
            break
        if matrix[i,j] != matrix[i,j-1]:
            i  = i - weights[j-1]
            value += values[j-1]
            taken[j-1] = 1  
        j -= 1
        
    print '  >> max value: ', value
  
    '''
    # DBG
    print values
    print weights
    print capacity
    print matrix
    '''
                
    return {"taken":taken,"value":value}



def greedy_solveIt(values,weights,capacity):
   
    items = len(values)

    # a trivial greedy algorithm for filling the knapsack
    # it takes items in-order until the knapsack is full
    value = 0
    weight = 0
    taken = []

    for i in range(0, items):
        if weight + weights[i] <= capacity:
            taken.append(1)
            value += values[i]
            weight += weights[i]
        else:
            taken.append(0)

    return {"taken":taken,"value":value}


def solveIt(inputData):

    # parse the input
    lines = inputData.split('\n')
    
    firstLine = lines[0].split()
    items = int(firstLine[0])
    capacity = int(firstLine[1])

    values = []
    weights = []

    for i in range(1, items+1):
        line = lines[i]
        parts = line.split()

        values.append(int(parts[0]))
        weights.append(int(parts[1]))

    # ========================================
    out = bb_solveIt(values,weights,capacity)
    #out = dp_solveIt(values,weights,capacity)
    # ========================================

    # prepare the solution in the specified output format
    outputData = str(out["value"]) + ' ' + str(1) + '\n'
    outputData += ' '.join(map(str, out["taken"]))
    return outputData


import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        fileLocation = sys.argv[1].strip()
        inputDataFile = open(fileLocation, 'r')
        inputData = ''.join(inputDataFile.readlines())
        inputDataFile.close()
        print solveIt(inputData)
    else:
        print 'This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)'

