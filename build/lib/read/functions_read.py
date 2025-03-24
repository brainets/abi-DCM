import scipy.io as sio
import numpy

def read_subject_data(data_path=None, subject=None, onset_time=False):
    
    data = sio.loadmat(data_path)

    time_pts = data['time'].T
    if onset_time:
        onset_ind = numpy.where(time_pts==0.)[0][0]
    else:
        onset_ind=0
        
    # Choose one subject's data
    subject_ind = numpy.where(data['subject']==subject)[0]
    
    # Cortical data of that subject
    subject_data = data['data'][subject_ind][:,:82,:]
 
    return time_pts[onset_ind:], subject_data[..., onset_ind:]