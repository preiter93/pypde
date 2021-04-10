def set_bc(D,B,pos):
    ''' Replace rows of D with B to apply BCs '''
    assert D.shape[0] == D.shape[1]
    assert D.shape == B.shape
    
    if not isinstance(pos, list): 
        pos = [pos]
        
    for p in pos: 
        D[p,:] = B[p,:] # replace