#pop_coupling.py has modules to calculate the z-scored population coupling matrix for all cells and visual stimuli
# in a given container ID.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance



def get_sessions(boc, container_id):
    # get ophys experiments from requested experiment container
    print(type(container_id))
    expt_session_info = pd.DataFrame(boc.get_ophys_experiments(experiment_container_ids=[container_id]))
    print('Experiment container info:')
    print(boc.get_experiment_containers(ids=[container_id]))

    # Create list of 3 session IDs in exp container, in standard order.
    container_session_ids = [expt_session_info[expt_session_info['session_type'] == 'three_session_A']['id'].values[0],
                             expt_session_info[expt_session_info['session_type'] == 'three_session_B']['id'].values[0],
                             expt_session_info[expt_session_info['session_type'] == 'three_session_C']['id'].values[0]]

    return expt_session_info, container_session_ids


def get_data_sets(boc, container_session_ids):
    # -Create data_set object for each session, place them in a list
    # -Get specimen ids for each session, put arrays in list
    data_sets = []
    specimens_by_session = []
    for i in range(3):
        data_sets.append(boc.get_ophys_experiment_data(ophys_experiment_id=container_session_ids[i]))
        specimens_by_session.append(data_sets[i].get_cell_specimen_ids())
    specimens_by_session = np.array(specimens_by_session)
    return data_sets, specimens_by_session


def get_traces(units, current_data_set, stable_specimen_indices, trace_type):
    # Get raw data
    if trace_type == 'corrected':
        timestamps, traces = current_data_set.get_corrected_fluorescence_traces()
    elif trace_type == 'dff':
        timestamps, traces = current_data_set.get_dff_traces()
    elif trace_type == 'raw':
        timestamps, traces = current_data_set.get_fluorescence_traces()
    elif trace_type == 'neuropil':
        timestamps, traces = current_data_set.get_neuropil_traces()

    # Filter (or not) for units that are stable across all 3 sessions
    if units == 'all':
        pass
    elif units == 'stable':
        traces = traces[stable_specimen_indices, :]

    return timestamps, traces



def get_activity_matrix(boc, container_id, session_idx, stim_type, trace_type):
    """
    #This function returns the activity_mtarix (ncellsxtime), which is the calcium traces across time
    :param container_id: experiment container ID
    :param session_idx: either 0, 1, or 2
    :param stim_type: name of stim
    :param trace_type: either 'dff' or 'corrected'. 'dff' is performed on the corrected traces
    :return: activity_matrix for a particular container_id, session, and stim_type
    """

    expt_session_info, container_session_ids = get_sessions(boc, container_id)
    stim_list = boc.get_ophys_experiment_data(ophys_experiment_id=container_session_ids[session_idx]).list_stimuli()

    #give separate names to different repeats of the same stimulus that are run in the same session

    #sontaneous shown twice in session C
    if 'spontaneous' in stim_list:
        stim_list.append(unicode('spontaneous_1'))
        stim_list.append(unicode('spontaneous_2'))

    # drifting grating is shown three times (session A)
    if 'drifting_gratings' in stim_list:
        stim_list.append(unicode('drifting_gratings_1'))
        stim_list.append(unicode('drifting_gratings_2'))
        stim_list.append(unicode('drifting_gratings_3'))

    # natural scenes is shown three times (session B)
    if 'natural_scenes' in stim_list:
        stim_list.append(unicode('natural_scenes_1'))
        stim_list.append(unicode('natural_scenes_2'))
        stim_list.append(unicode('natural_scenes_3'))

    #there are three natural movies. natural movie three is shown twice (session A)
    if 'natural_movie_three' in stim_list:
        stim_list.append(unicode('natural_movie_three_1'))
        stim_list.append(unicode('natural_movie_three_2'))

    # static gratings is shown three times (session B)
    if 'static_gratings' in stim_list:
        stim_list.append(unicode('static_gratings_1'))
        stim_list.append(unicode('static_gratings_2'))
        stim_list.append(unicode('static_gratings_3'))

    # locally sparse noise is shown three times (session C)
    if 'locally_sparse_noise' in stim_list:
        stim_list.append(unicode('locally_sparse_noise_1'))
        stim_list.append(unicode('locally_sparse_noise_2'))
        stim_list.append(unicode('locally_sparse_noise_3'))

    if stim_type not in stim_list:
        raise ValueError('Requested stim_type is not present in the stimulus names for requested session.')

    data_sets, specimens_by_session = get_data_sets(boc, container_session_ids)
    current_data_set = data_sets[session_idx]

    # Find cell specimens present in all 3 sessions
    stable_specimen_ids = set(specimens_by_session[0]) & set(specimens_by_session[1]) & set(specimens_by_session[2])
    stable_specimen_ids = np.array(list(stable_specimen_ids))
    stable_specimen_indices = current_data_set.get_cell_specimen_indices(stable_specimen_ids)

    timestamps, traces = get_traces('all', current_data_set, stable_specimen_indices, trace_type)

    # get requested activity_matrix
    if stim_type[:11] == 'spontaneous':
        stim_table = current_data_set.get_stimulus_table('spontaneous')
        if stim_type == 'spontaneous_1':
            activity_matrix = traces[:, stim_table.start[0]: stim_table.end[0]]
        elif stim_type == 'spontaneous_2':
            activity_matrix = traces[:, stim_table.start[1]: stim_table.end[1]]

    elif stim_type[:17] == 'drifting_gratings':
        stim_table = current_data_set.get_stimulus_table('drifting_gratings')

        #separate the blocks
        start1_ind = 0
        end1_ind = 199
        start2_ind = 200
        end2_ind = 399
        start3_ind = 400
        end3_ind = 599
        if (stim_table.end[end1_ind]-stim_table.start[start1_ind]>19000) or \
                (stim_table.end[end2_ind]-stim_table.start[start2_ind]>19000) or\
                (stim_table.end[end3_ind]-stim_table.start[start3_ind]>19000):

            raise ValueError('The start and end indices for one of the blocks is wrong!!')

        if stim_type == 'drifting_gratings_1':
            activity_matrix = traces[:, stim_table.start[start1_ind]: stim_table.end[end1_ind]]
        elif stim_type == 'drifting_gratings_2':
            activity_matrix = traces[:, stim_table.start[start2_ind]: stim_table.end[end2_ind]]
        elif stim_type == 'drifting_gratings_3':
            activity_matrix = traces[:, stim_table.start[start3_ind]: stim_table.end[end3_ind]]



    #pull out three blocks of natural scences
    elif stim_type[:-2] == 'natural_scenes':
        stim_table = current_data_set.get_stimulus_table('natural_scenes')

        #find the indices of each block
        diff_start=np.diff(stim_table.start)
        sort_index=np.argsort(diff_start)[::-1]
        sorted_diff_start=diff_start[sort_index]
        start1_ind=0
        end1_ind=min(sort_index[0:2])
        start2_ind=end1_ind+1
        end2_ind = max(sort_index[0:2])
        start3_ind=end2_ind+1
        end3_ind=len(stim_table.start)-1

        #if the duration of any of the above blocks is greater than 9 minutes (14483 frames), something went wrong
        if (stim_table.end[end1_ind]-stim_table.start[start1_ind]>16300) or \
                (stim_table.end[end2_ind]-stim_table.start[start2_ind]>16300) or\
                (stim_table.end[end3_ind]-stim_table.start[start3_ind]>16300):
            raise ValueError('The start and end indices for one of the blocks is wrong!!')

        if stim_type == 'natural_scenes_1':
            activity_matrix = traces[:, stim_table.start[start1_ind]: stim_table.end[end1_ind]]
        elif stim_type == 'natural_scenes_2':
            activity_matrix = traces[:, stim_table.start[start2_ind]: stim_table.end.values[end2_ind]]
        elif stim_type == 'natural_scenes_3':
            activity_matrix = traces[:, stim_table.start[start3_ind]: stim_table.end.values[end3_ind]]

    elif stim_type[:13] == 'natural_movie':

        if stim_type[:17] == 'natural_movie_one':
            stim_table = current_data_set.get_stimulus_table(stim_type) #movies one and two don't have repeats
            activity_matrix = traces[:, stim_table.start[0]: stim_table.end.values[-1]]

        elif stim_type[:17] == 'natural_movie_two':
            stim_table = current_data_set.get_stimulus_table(stim_type) #movies one and two don't have repeats
            activity_matrix = traces[:, stim_table.start[0]: stim_table.end.values[-1]]

        elif stim_type[:-2]=='natural_movie_three':
            stim_table = current_data_set.get_stimulus_table(stim_type[:-2])  # index to drop '_2' in 'natural_movie_three_2'
            # separate the blocks
            start1_ind = 0
            end1_ind = 17999
            start2_ind = 18000
            end2_ind = 35999
            if (stim_table.end[end1_ind] - stim_table.start[start1_ind] > 19000) or \
                    (stim_table.end[end2_ind] - stim_table.start[start2_ind] > 19000):
                raise ValueError('The start and end indices for one of the blocks is wrong!!')

            if stim_type == 'natural_movie_three_1':
                activity_matrix = traces[:, stim_table.start[start1_ind]: stim_table.end[end1_ind]]

            elif stim_type == 'natural_movie_three_2':
                activity_matrix = traces[:, stim_table.start[start2_ind]: stim_table.end.values[end2_ind]]

    elif stim_type[:-2] == 'static_gratings':
        stim_table = current_data_set.get_stimulus_table('static_gratings')

        #find the indices of each block
        diff_start=np.diff(stim_table.start)
        sort_index=np.argsort(diff_start)[::-1]
        sorted_diff_start=diff_start[sort_index]
        #the top two indices will now be the start_index of the third and second block. the third block comes 14 minutes
        #after the last presentation of the second trial, so it has the largest inter-presentation difference.
        start1_ind=0
        end1_ind=min(sort_index[0:2])
        start2_ind=end1_ind+1
        end2_ind = max(sort_index[0:2])
        start3_ind=end2_ind+1
        end3_ind=len(stim_table.start)-1

        #if the duration of any of the above blocks is greater than 9 minutes (16290 frames), something went wrong
        if (stim_table.end[end1_ind]-stim_table.start[start1_ind]>16200) or \
                (stim_table.end[end2_ind]-stim_table.start[start2_ind]>16200) or\
                (stim_table.end[end3_ind]-stim_table.start[start3_ind]>16400):
            raise ValueError('The start and end indices for one of the blocks is wrong!!')

        if stim_type == 'static_gratings_1':
            activity_matrix = traces[:, stim_table.start[start1_ind]: stim_table.end[end1_ind]]
        elif stim_type == 'static_gratings_2':
            activity_matrix = traces[:, stim_table.start[start2_ind]: stim_table.end.values[end2_ind]]

        elif stim_type == 'static_gratings_3':
            activity_matrix = traces[:, stim_table.start[start3_ind]: stim_table.end.values[end3_ind]]



    elif stim_type[:20] == 'locally_sparse_noise':
        stim_table = current_data_set.get_stimulus_table('locally_sparse_noise')

        #find the indices of each block
        diff_start=np.diff(stim_table.start)
        sort_index=np.argsort(diff_start)[::-1]
        sorted_diff_start=diff_start[sort_index]
        #the top two indices will now be the start_index of the third and second block. the third block comes 14 minutes
        #after the last presentation of the second trial, so it has the largest inter-presentation difference.
        start1_ind=0
        end1_ind=min(sort_index[0:2])
        start2_ind=end1_ind+1
        end2_ind = max(sort_index[0:2])
        start3_ind=end2_ind+1
        end3_ind=len(stim_table.start)-1

        #if the duration of any of the above blocks is greater than 9 minutes (2442 presentations), something went wrong
        if (stim_table.end[end1_ind]-stim_table.start[start1_ind]>23580) or \
                (stim_table.end[end2_ind]-stim_table.start[start2_ind]>23580) or\
                (stim_table.end[end3_ind]-stim_table.start[start3_ind]>23580):
            raise ValueError('The start and end indices for one of the blocks is wrong!!')

        if stim_type == 'locally_sparse_noise_1':
            activity_matrix = traces[:, stim_table.start[start1_ind]: stim_table.end[end1_ind]]
        elif stim_type == 'locally_sparse_noise_2':
            activity_matrix = traces[:, stim_table.start[start2_ind]: stim_table.end.values[end2_ind]]
        elif stim_type == 'locally_sparse_noise_3':
            activity_matrix = traces[:, stim_table.start[start3_ind]: stim_table.end.values[end3_ind]]

    else:
        print('Failed to create activity matrix, or experiment type not available')

    activity_matrix = activity_matrix.T
    return activity_matrix, stable_specimen_indices, stable_specimen_ids

def pop_corr(activity_matrix):
    """
    main function for population coupling metric. input is timeseriesxncells.
    output is 1xn population metric.
    time series with variable coupling (time, n_cells) are the input
    Author: Madineh, Phil & Max

    :param activity_matrix: Time series per cell
    :return: pop_corr_array: Population coupling per cell
    """
    activity_matrix_mean_adj = activity_matrix - activity_matrix.mean(axis=0)

    n_cells = activity_matrix.shape[1]
    # preallocate one matrix for averages

    sum_mean_adj = activity_matrix_mean_adj.sum(axis=1)
    pop_corr_array = np.zeros((n_cells, 1))

    for i in range(n_cells):
        cell_excluded_sum_mean_adj = sum_mean_adj - activity_matrix_mean_adj[:, i]

        pop_corr_array[i] = np.sum(activity_matrix[:, i] * cell_excluded_sum_mean_adj) \
                            / activity_matrix[:, i].std()

    return pop_corr_array

def pop_corr_z_scored(activity_matrix):
    """
    return mean population coupling divided by sample standard deviation
    Author: Max

    :param activity_matrix: Time series per cell
    :return: pop_corr_array: Z-scored population coupling per cell
    """
    pop_array = pop_corr(activity_matrix)
    return (pop_array-pop_array.mean())/pop_array.std(ddof=1)

def get_matrices_all(boc, container_id,trace_type):
    """
    Get all the pop coupling matrices (plus a few other variables) for a given container_id
    :param container_id:
    :return: pc_mat, pcz_mat, mean_act_mat, distance_mat, stable_specimen_indices
    pc_mat is the matrix of pop coupling (raw) for all stimuli in all sessions: ncellsxnstim
    pcz_mat is the same, but z-scored ncellsxnstim
    mean_act_mat is the mean activity (dff) for each stim across all sessions ncellsx1
    distance_mat is the distance to center of frame for each cell
    stable_specimen_indices is the indices of cells that were stable across all sessions
    note: The population for each stimulus is all firing cells. After the calculation, the pop coupling is only kept
    for cells stable across all three sessions.
    """

    #get the activity matrices
    session_ids=[0,1,2]
    stim_names_A=['drifting_gratings_1', 'natural_movie_three_1','natural_movie_one','drifting_gratings_2',
                  'spontaneous_1','natural_movie_three_2','drifting_gratings_3']
    stim_names_B=['static_gratings_1', 'natural_scenes_1','spontaneous_1','natural_scenes_2',
                  'static_gratings_2','natural_movie_one','natural_scenes_3','static_gratings_3']
    stim_names_C=['locally_sparse_noise_1','spontaneous_1','natural_movie_one','locally_sparse_noise_2',
                  'natural_movie_two', 'spontaneous_2','locally_sparse_noise_3']

    stim_names=[]
    pcz_mat=[]
    pc_mat=[]
    mean_act_mat=[]
    mean_run_vect=[]

    for session_idx in session_ids:
        if session_idx==0:
            stim_names=stim_names_A
        elif session_idx==1:
            stim_names=stim_names_B
        elif session_idx==2:
            stim_names=stim_names_C

        for ind,stim_name in enumerate(stim_names):
            print stim_name
            activity_matrix, stable_specimen_indices, stable_specimen_ids=\
                get_activity_matrix(boc, container_id, session_idx, stim_name, trace_type=trace_type)

            mean_activity = np.mean(activity_matrix[:,stable_specimen_indices,],axis=0)
            mean_act_mat.append(mean_activity)

            pcz = pop_corr_z_scored(activity_matrix)[stable_specimen_indices, :]
            pcz_mat.append(pcz)

    pcz_mat = np.squeeze(np.array(pcz_mat).T)

    return pcz_mat

def get_invariance_bound_scaled(pcz_mat):
    """

    :param pcz_mat: matrix of z-scored pop coupling for all stimuli in all sessions (cellsxstimuli)
    :return: single scalar [0-1] that measures the invariance of pop coupling to stimulus
    """

    # sort the fucking pcz
    mean_pcz = np.mean(pcz_mat, axis=1)
    sort_ind = np.argsort(mean_pcz)
    pcz_mat = pcz_mat[sort_ind, :]

    num_cells = np.shape(pcz_mat)[0]
    num_stims = np.shape(pcz_mat)[1]

   # get the lower bound of invariance index by shuffling each column (shared stim)
    pcz_mat_lb = np.copy(pcz_mat)
    for i in range(num_stims):
        rand_order = np.random.permutation(num_cells)
        pcz_mat_lb[:,i] = pcz_mat[rand_order,i]

    # get the upper bound of invariance index by sorting each column (shared stim) separately - this will redue cell variance
    pcz_mat_ub = np.copy(pcz_mat)
    for i in range(num_stims):
        sort_order = np.argsort(pcz_mat[:, i])
        pcz_mat_ub[:, i] = pcz_mat[sort_order, i]


    Aflat = pcz_mat.flatten()
    LBflat = pcz_mat_lb.flatten()
    UBflat = pcz_mat_ub.flatten()
    dist_lb = distance.euclidean(Aflat, LBflat)
    dist_ub = distance.euclidean(Aflat, UBflat)
    index = dist_lb / (dist_lb + dist_ub)

    return index

def plot_pcz_matrix(pcz_mat,container_id):

    num_stims = np.shape(pcz_mat)[1]
    num_cells = np.shape(pcz_mat)[0]
    mean_pcz = np.mean(pcz_mat, axis=1)

    # sort the rows (cells) by average population coupling across all stimulus
    sort_ind = np.argsort(mean_pcz)
    fig, axs = plt.subplots(figsize=(10, 8.5))
    ax = sns.heatmap(pcz_mat[sort_ind, :])

    # Add a label to the colorbar
    colorbar = ax.collections[0].colorbar
    colorbar.set_label('Pop Coupling (Z scored)', rotation=270, labelpad=25)

    # Label the stim names on the bottom x-axiz
    xlabels = ['DG', 'NM_3', 'NM_1', 'DG', 'S', 'NM_3', 'DG', 'SG', 'NS', 'S', 'NS', 'SG', 'NM_1', 'NS', 'SG', 'LSN',
               'S', 'NM_1', 'LSN',
               'NM_2', 'S', 'LSN']
    plt.xticks(0.5 + np.arange(num_stims), xlabels, rotation='vertical')
    plt.xlabel('Visual Stimulus', labelpad=15)
    plt.xlim([0, num_stims])
    plt.gcf().subplots_adjust(bottom=0.15)  # add some space for xlabel

    # Label the sessions on the top of x-axis
    plt.annotate('Session A', (0.18, 0.925), xycoords='figure fraction', textcoords='offset points', va='top')
    plt.annotate('Session B', (0.40, 0.925), xycoords='figure fraction', textcoords='offset points', va='top')
    plt.annotate('Session C', (0.6, 0.925), xycoords='figure fraction', textcoords='offset points', va='top')

    # and add vertical lines to denote sessions
    plt.axvline(x=7)
    plt.axvline(x=15)

    # Set the  y axes labels
    ylabels = [i for i in np.arange(0, num_cells, 10)]
    plt.yticks(np.arange(0, num_cells, 10), ylabels)
    plt.ylabel('Cell Number', labelpad=10)
    invariance_ind = get_invariance_bound_scaled(pcz_mat)
    plt.suptitle('PCZ matrix for container ID : %.2f' % container_id)
    plt.show()
    return
