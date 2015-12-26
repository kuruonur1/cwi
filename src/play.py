import numpy as np

if __name__ == '__main__':
    out_mask = np.array([[1,1,0,0],[1,1,1,0]])
    target_output = np.array([[0,1,0,0],[1,0,1,0]])
    # output = np.array([[.2,.1,0,0],[.6,.4,.7,0]]) + np.finfo(float).eps
    output = np.array([[.2,.1,0,0],[.6,.4,.7,0]])
    print 'out_mask:'
    print out_mask
    print 'target_output:'
    print target_output
    print 'output:'
    print output
    print 'output[out_mask]:'
    print output[out_mask.nonzero()]
    print 'target_output[out_mask]:'
    print target_output[out_mask]
    # print -np.sum(out_mask*target_output*np.log(output) + out_mask*(1-target_output)*np.log(1-output)) / np.sum(out_mask)
