def set_bc(D,B,pos):
    ''' Replace rows of D with B to apply BCs '''
    assert D.shape[0] == D.shape[1]
    assert D.shape == B.shape
    
    if not isinstance(pos, list): 
        pos = [pos]
        
    for p in pos: 
        D[p,:] = B[p,:] # replace


def pad(array, reference, offsets):
    """
    array: Array to be padded
    reference: Reference array with the desired shape
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    """
    # Create an array of zeros with the reference shape
    result = np.zeros(reference.shape)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offset[dim], offset[dim] + array.shape[dim]) for dim in range(a.ndim)]
    # Insert the array in the result at the specified offsets
    result[insertHere] = a
    return result