'''
Module - simpy  	version: 3.0
--------------------------------
Module for creation of artificial images with certain values of the Minkowski functionals
Also allows to optimize a previous image to the target values

Functions included in this module:
    - imaGen(): generates random image (array) of given M0 value (area/volume for 2D/3D respectively)
    - optimFun(): optimizes preexisting image (already with desired M0) to achieve the Mi target values of the rest of functionals
    - optimDir(): optimization through directed method
    - minmodel(): allows direct simulation of images
    - downscaling(): reduces resolution of an image

For more information see docstrings for each function
'''

import minpy as mp
import numpy as np
import random



def imaGen (dim, target_M0):
    '''
    Returns random array of given dimensions and M0 functional value

    Inputs:
        - dim: tuple ((Lz),Ly,Lx) of image dimensions
        - target_M0: desired minkowski functional of order 0 (either area or volume)
    Output: numpy array of given dimensions with random values that attain desired M0
    '''
    array = np.zeros(dim, dtype=int)
    curr_M0 = 0
    #Until target M0 is achieved, random positions are generated and their value changed to 1
    while curr_M0 != target_M0:
        if len(dim) == 2:
            pos = (random.randint(0,dim[0]-1), random.randint(0,dim[1]-1))
        elif len(dim) == 3:
            pos = (random.randint(0,dim[0]-1),random.randint(0,dim[1]-1), random.randint(0,dim[2]-1))
        if array[pos] == 0:
            array[pos] = 1
            curr_M0 += 1
    return array

#end imaGen()



def optimFun (input_array, target_fun, verbose=False, k=2.5, s=1.15, iter_limit=np.inf):
    '''
    Optimizes array to target minkowski functional values

    Input:
        - input_array: numpy binary array (0 = background; 1 = foreground). It must have desired M0
        - target_fun: list [M0, ..., Mn] containing desired minkowski functionals

        - verbose: boolean determining whether current functionals are periodically printed
        - k, s: constants involving selection of the changes during optimization (see code)
        - iter_limit: maximum nº of iterations accepted in the optimization process (to limit the time it can take)

    Output: numpy array representing image with desired minkowski functionals

    WARNING: ONLY experimental (-> attainable) combinations of functionals should be used as input.
    Non experimental functionals may be unnatainable and thus program may enter infinite loop
    '''
    if input_array.ndim == 2:
    #Code for 2D images
        #Let the coordinates of input_array be (y,x), so that 0 <= x <= LX ; 0 <= y <= LY
        #Then, "array" is an array for which -1 <= x <= LX+1 ; -1 <= y <= LY
        #That is, array has one more lenght in each direction (+/- x and +/- y)
        #This is necessary for the way functionals are updated
        LY, LX = np.array(np.shape(input_array)) + [2,2]
        array = np.zeros((LY, LX))
        array[1:LY-1, 1:LX-1] = input_array #The central part of array corresponds to the input_array
        curr_fun = mp.fun(array)

        if curr_fun[0] != target_fun[0]:
            raise Exception('WARNING: input array\'s M0 doesn\'t match target M0. \nSee docstring for more info on usage.')

        iter = 0 #Variable containing number of iterations. If surpasses iter_limit process is finished
        while curr_fun != target_fun and iter <= iter_limit: #Each iteration a pair of points p,q have its value swapped if that improves the image
            iter += 1
            py, px = (random.randint(1, LY-2), random.randint(1, LX-2))
            qy, qx = (random.randint(1, LY-2), random.randint(1, LX-2))
            if array[py,px] != array[qy, qx]: #if the 2 points have the same value swapping them is useless
                #New Mi values are computed by updating, measuring the change in the 8-adyacent region produced by
                #Changing p value to q value, and then changing q value to (previous) p value
                funP_ini = mp.fun(array[py-1:py+2,px-1:px+2])
                array[py, px] = not array[py, px]
                funP_fin = mp.fun(array[py-1:py+2,px-1:px+2])
                funQ_ini = mp.fun(array[qy-1:qy+2,qx-1:qx+2])
                array[qy,qx] = not array[qy,qx]
                funQ_fin = mp.fun(array[qy-1:qy+2,qx-1:qx+2])
                #Change of functionals + previous functionals -> new_functionals
                fun_change = np.array(funP_fin) - np.array(funP_ini) + np.array(funQ_fin) - np.array(funQ_ini)
                new_curr_fun = list(np.array(curr_fun) + fun_change)
                #Improves in Mi represent how farther were previous functionals from target than new funcitonals
                #In other words, how nearer are newer_functionals
                improve_M1 = abs(curr_fun[1]-target_fun[1]) - abs(new_curr_fun[1]-target_fun[1])
                improve_M2 = abs(curr_fun[2]-target_fun[2]) - abs(new_curr_fun[2]-target_fun[2])
                #Changes are accepted if they improve or maintain the distance to target_funs
                #The different functionals have different speeds of change. k constant makes changes in M2 comparable to changes in M1
                accepted = improve_M1 + k*improve_M2 >= 0
                if verbose and iter%5000 == 0: #Only prints every 5000 iterations
                    print(curr_fun)
                if accepted:
                    curr_fun = new_curr_fun
                else:   #If changes are not accepted they are reverted
                    array[py, px] = not array[py, px]
                    array[qy,qx] = not array[qy,qx]
        return array[1:LY-1, 1:LX-1]

    elif input_array.ndim == 3:
    #Code for 3D images. Most of code is analogous to 2D (see 2D comments)
        LZ, LY, LX = np.array(np.shape(input_array)) + [2,2,2]
        array = np.zeros((LZ, LY, LX))
        array[1:LZ-1,1:LY-1, 1:LX-1] = input_array
        curr_fun = mp.fun(array)

        iter = 0
        while curr_fun != target_fun and iter <= iter_limit:
            iter+=1
            pz, py, px = (random.randint(1, LZ-2), random.randint(1, LY-2), random.randint(1, LX-2))
            qz, qy, qx = (random.randint(1, LZ-2), random.randint(1, LY-2), random.randint(1, LX-2))
            if array[pz, py, px] != array[qz, qy, qx]:
                funP_ini = mp.fun(array[pz-1:pz+2,py-1:py+2,px-1:px+2])
                array[pz, py, px] = not array[pz, py, px]
                funP_fin = mp.fun(array[pz-1:pz+2,py-1:py+2,px-1:px+2])
                funQ_ini = mp.fun(array[qz-1:qz+2,qy-1:qy+2,qx-1:qx+2])
                array[qz, qy, qx] = not array[qz, qy, qx]
                funQ_fin = mp.fun(array[qz-1:qz+2,qy-1:qy+2,qx-1:qx+2])
                fun_change = np.array(funP_fin) - np.array(funP_ini) + np.array(funQ_fin) - np.array(funQ_ini)
                new_curr_fun = list(np.array(curr_fun) + fun_change)
                improve_M1 = (abs(curr_fun[1]-target_fun[1]) - abs(new_curr_fun[1]-target_fun[1]))
                improve_M2 = (abs(curr_fun[2]-target_fun[2]) - abs(new_curr_fun[2]-target_fun[2]))
                improve_M3 = (abs(curr_fun[3]-target_fun[3]) - abs(new_curr_fun[3]-target_fun[3]))
                accepted = improve_M1 + s*improve_M2 + k*improve_M3 >= 0 #Again s and k values make changes in Mi comparable to each other
                if verbose and iter%5000 == 0:
                    print(curr_fun)
                if accepted:
                    curr_fun = new_curr_fun
                else:
                    array[pz,py,px] = not array[pz,py,px]
                    array[qz,qy,qx] = not array[qz,qy,qx]
        return array[1:LZ-1, 1:LY-1, 1:LX-1]

#end optimFun()



def optimDir (input_array, target_fun, verbose=False, k=2.5, s=1.15):
    '''
    Optimizes array to target Minkowski functional values through a directed method
    WARNING: directed method is actually slower and susceptible to get stuck in relative minima

    Use:
        - input_array: numpy binary array (0 = background; 1 = foreground). It must have desired M0
        - target_fun: list [M0, ..., Mn] containing desired minkowski functionals

        - k, s: constants involving selection of the changes during optimization (see code)

    Output: numpy array representing image with desired minkowski functionals
    '''
    if input_array.ndim == 2:
    #Code for 2D images
        #Array is made with 2 more length in each direction (see optimFun comments)
        #This is necessary for the way Mi changes for each position are optimizationupdated
        LY, LX = np.array(input_array.shape) + [4,4]
        array = np.zeros((LY, LX))
        array[2:LY-2, 2:LX-2] = input_array
        curr_fun = mp.fun(array)
        fun_array = np.zeros((LY, LX, 2))
        #fun_array keeps the Mi change resulting from swapping value 0 <-> 1 in a given point
        #Initially this is measured for each point (corresponding to the zone of input_array)
        for y in range(2,LY-2):
            for x in range(2,LX-2):
                #A copy of the 8-adyacent region is made to measure the change in Mi
                window = np.copy(array[y-1:y+2,x-1:x+2])
                fun_array[y,x] = mp.fun(window)[1:]
                window[1,1] = not window[1,1]
                fun_array[y,x] = np.array(mp.fun(window)[1:]) - fun_array[y,x]
        #Optimization is made iteratively by changing values 0 -> 1 and 1 -> 0 alternatively
        #This is necessary since making both changes at the same time leads to problems when both points are 8-adyacent
        turn = 0
        while curr_fun != target_fun:
            potential = [0] #Variable that will hold [0] if no good change is found and [improve, y, x] if one is found
            for y in range(2,LY-2):
                for x in range(2,LX-2):
                    if turn == array[y,x]: #Only positions with appropiate value are checked for
                        new_fun = list(np.array(curr_fun[1:]) + fun_array[y,x])
                        improve_M1 = abs(target_fun[1] - curr_fun[1]) - abs(target_fun[1] - new_fun[0])
                        improve_M2 = abs(target_fun[2] - curr_fun[2]) - abs(target_fun[2] - new_fun[1])
                        net_improve = improve_M1 + improve_M2*k
                        #If the change produces a net improve and its bigger than the latest best improvement
                        #It is set as the new best improvement
                        if net_improve > 0 and net_improve > potential[0]:
                            potential = [net_improve,y,x]

            if len(potential) > 1: #Potential has been updated, i.e. a good change was found
                y,x = potential[1:]
            elif len(potential) == 1: #A good change was not found -> random change (better than taking the least worse change, it would make it get stuck)
                y,x = (random.randint(2, LY-3), random.randint(2, LX-3))
                while array[y,x] != turn: #Random positions are made until one has the value its supposed to have
                    y,x = (random.randint(2, LY-3), random.randint(2, LX-3))
            turn = not turn #turn is changed for next iteration
            #fun array contains the fun change so this way curr fun is updated
            curr_fun[1] += fun_array[y,x,0]
            curr_fun[2] += fun_array[y,x,1]
            array[y,x] = not array[y,x]
            #Finally, fun_array is updated (IMPORTANT)
            #Changing the value of point p means all its 8-adyacent points now have a different Mi change, so it must be recalculated
            #The 8-adyacent region of a point in the boundarie of input_array includes positions "outbounding" that array
            #Thus, 2 extra units of length were added in each direction so as to not have out of boundarie problems
            for y2 in range(y-1,y+2):
                for x2 in range(x-1,x+2):
                    window = np.copy(array[y2-1:y2+2,x2-1:x2+2])
                    fun_array[y2,x2] = mp.fun(window)[1:]
                    window[1,1] = not window[1,1]
                    fun_array[y2,x2] = np.array(mp.fun(window)[1:]) - fun_array[y2,x2]
        return array[2:LY-2, 2:LX-2]

    elif input_array.ndim == 3:
    #Code for 3D images. Mostly analogous to 2D case (see 2D comments)
        LZ, LY, LX = np.array(input_array.shape) + [4,4,4]
        array = np.zeros((LZ, LY, LX))
        array[2:LZ-2,2:LY-2, 2:LX-2] = input_array
        curr_fun = mp.fun(array)
        fun_array = np.zeros((LZ, LY, LX, 3))
        for z in range(2,LZ-2):
            for y in range(2,LY-2):
                for x in range(2,LX-2):
                    window = np.copy(array[z-1:z+2,y-1:y+2,x-1:x+2])
                    fun_array[z,y,x] = mp.fun(window)[1:]
                    window[1,1,1] = not window[1,1,1]
                    fun_array[z,y,x] = np.array(mp.fun(window)[1:]) - fun_array[z,y,x]
        turn = 0
        while curr_fun != target_fun:
            if verbose == True:
                print(curr_fun)
            potential = [0]
            for z in range(2,LZ-2):
                for y in range(2,LY-2):
                    for x in range(2,LX-2):
                        new_fun = list(np.array(curr_fun[1:]) + fun_array[z,y,x])
                        improve_M1 = abs(target_fun[1] - curr_fun[1]) - abs(target_fun[1] - new_fun[0])
                        improve_M2 = abs(target_fun[2] - curr_fun[2]) - abs(target_fun[2] - new_fun[1])
                        improve_M3 = abs(target_fun[3] - curr_fun[3]) - abs(target_fun[3] - new_fun[2])
                        net_improve = improve_M1 + improve_M2*s + improve_M3*k
                        if net_improve > 0 and net_improve > potential[0] and turn == array[z,y,x]:
                            potential = [net_improve,z,y,x]


            if len(potential) > 1:
                z,y,x = potential[1:]
            elif len(potential) == 1:
                z,y,x = (random.randint(2, LZ-3), random.randint(2, LY-3), random.randint(2, LX-3))
                while array[z,y,x] != turn:
                    z,y,x = (random.randint(2, LZ-3), random.randint(2, LY-3), random.randint(2, LX-3))
            turn = not turn
            curr_fun[1] += fun_array[z,y,x,0]
            curr_fun[2] += fun_array[z,y,x,1]
            curr_fun[3] += fun_array[z,y,x,2]
            array[z,y,x] = not array[z,y,x]
            for z2 in range(z-1,z+2):
                for y2 in range(y-1,y+2):
                    for x2 in range(x-1,x+2):
                        window = np.copy(array[z2-1:z2+2,y2-1:y2+2,x2-1:x2+2])
                        fun_array[z2,y2,x2] = mp.fun(window)[1:]
                        window[1,1,1] = not window[1,1,1]
                        fun_array[z2,y2,x2] = np.array(mp.fun(window)[1:]) - fun_array[z2,y2,x2]
        return array[2:LZ-1,2:LY-2, 2:LX-2]

#end optimDir



def minmodel (dim, target_fun, iter_limit=np.inf, end_directed=False, verbose=False):
    '''
    Function for directly simulating an image de novo with desired Mi values through the rest of module functions

    Input:
        - dim: tuple ((Lz),Ly,Lx) of image dimensions
        - target_fun: list [M0, ..., Mn] containing desired minkowski functionals

        - iter_limit: maximum nº of iterations accepted in the optimization process (to limit the time it can take)
        - end_directed: (True/False). If iter has been limited, imperfect image can be fastly adjusted to desired functionals (NOT recommended, see optimDir docstring)
        - verbose: boolean determining whether current functionals are periodically printed

    Output: numpy array representing image with desired minkowski functionals
    '''
    model = imaGen(dim, target_fun[0])
    result = optimFun(model, target_fun, verbose=verbose, iter_limit=iter_limit)
    if end_directed:
        result = optimDir(result, target_fun)
    return result

#end minmodel



def downscale (array, target_dims):
    '''
    Reduces the resolution of an array of certain dimensions
    to lower dimensions (as long as they are divisible!!) by clustering pixels together.

    Input:
        - array: numpy array to be downscaled
        - target_dims: tuple of dimension lengths to which downscale

    Output: downscaled array
    '''
    target_dims = np.array(target_dims)
    input_dims = array.shape

    #Relation between input and target dimension which is the stride neccesary
    #to iterate over the cluster of pixels that is reduced to a single pixel in
    #the output
    stride = (input_dims/target_dims).astype(int)

    output_array = np.zeros(target_dims)

    if len(input_dims) == 2:
        for y in range(0, input_dims[0], stride[0]):
            for x in range(0, input_dims[1], stride[1]):
                #Iteration over the cluster of pixels that are reduced to
                #a single pixel in outcome
                mean = np.mean(array[y:y+stride[0], x:x+stride[1]])
                #If the mean value of this cluster is >0.5 its reduced
                #to a 1 value, if it is <0.5 to a 0 value
                if mean >= 0.5:
                    output_array[y//stride[0], x//stride[1]] = 1
                else:
                    output_array[y//stride[0], x//stride[1]] = 0

    elif len(input_dims) == 3:
        for z in range(0, input_dims[0], stride[0]):
            for y in range(0, input_dims[1], stride[1]):
                for x in range(0, input_dims[2], stride[2]):
                    #Iteration over the cluster of pixels that are reduced to
                    #a single pixel in outcome
                    mean = np.mean(array[z:z+stride[0], y:y+stride[1], x:x+stride[2]])
                    #If the mean value of this cluster is >0.5 its reduced
                    #to a 1 value, if it is <0.5 to a 0 value
                    if mean >= 0.5:
                        output_array[z//stride[0], y//stride[1], x//stride[2]] = 1
                    else:
                        output_array[z//stride[0], y//stride[1], x//stride[2]] = 0

    return output_array

#end downscale
