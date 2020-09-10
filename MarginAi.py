#!/usr/bin/env python
# Logan Mulroney
# trnaFast5HMMs.py
# Uses Adam Novak's yahmm.py and Jacob Schreiber's PyPore
# Adapted from Miten Jain's trnaHMMs.py

import sys, os, argparse, collections, time, itertools, string, subprocess, pysam, random

from Bio.Seq import Seq
from Bio.Alphabet import generic_dna, generic_rna
from Bio import SeqIO

import numpy as np
import scipy as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.style.use('ggplot')

import pysam
import colorsys
import pyximport
import pymysql
pymysql.install_as_MySQLdb()
pyximport.install(setup_args={'include_dirs':np.get_include()})
from yahmm import *
from PyPore.core import Segment
from PyPore.parsers import *
from Fast5Types import *
import seaborn as sns
###############################################################################
def parse_args(args):
    '''
    parses command line arguments
    '''
    parser = argparse.ArgumentParser(description ='Parses signalAlign output'
                                     ' directory into summery text file')

    parser.add_argument('--infile', '-if', type = str,
                        help = 'input file with paths to fastq files, sequence_summary.txt'
                                ' path to fast5 files, output directories, and the sample label'
                                ' use this option in place of those other options')

    parser.add_argument('--fastq', '-i', type = str,
                        help = 'fastq files for alignment and the base name'
                        'for the rest of the analysis pipeline')

    parser.add_argument('--label', '-l', type = str, default = 'Experiment',
                        help = 'experiment label for testing purposes')

    parser.add_argument('--pathToFast5', '-p', type = str,
                        default = None, help = 'Path to directory of fast5'
                        ' files to be analyzed')

    parser.add_argument('--outputDir', '-d', type = str,
                        default = "./", help = 'path to output directory')

    parser.add_argument('--sequencing_summary', '-s', type = str, 
                        default = '', help = 'file that maps read name to  '
                        'the read IDs')

    parser.add_argument('--random_sample', '-r', type = float, default = 0.0,
                        help = 'randomly sample either n reads or x proportion of the sample'
                        ' >1 will sample that number of reads, 0-0.999 will sample'
                        ' that proportion of the sample')

    parser.add_argument('--models', '-m', type = str,
                        default = None, help = 'Path to directory of models if a non-standard polyI models is desired')

    parser.add_argument('--threads', '-t', type = int, default = 1, 
                        help = 'number of threads to use for read segmenting default=1')

    return parser.parse_args()
###############################################################################
# Create all possible kmers
########################################################################

def kmer_current_map(file):
    # Read kmer current mean, sd, and time measurements
    kmer_current_dict = {}
    kmer_table = open(file, 'r')
    for line in kmer_table:
        line = line.strip().split('\t')
        key = line[0].strip()
        meanCurrent = float(line[1].strip())
        stdevCurrent = float(line[2].strip())
        if not key in kmer_current_dict.keys():
            kmer_current_dict[key] = 0
        kmer_current_dict[key] = [meanCurrent, stdevCurrent]

    kmer_table.close()

    return kmer_current_dict

########################################################################
# HMM Constructor class constructing input-specific HMMs
########################################################################

class HMM_Constructor():

    def __init__(self):
        pass

    def HMM_linear_model(self, kmer_list, kmer_current_dict, model_name=None):
        '''
        This HMM models the segments corresponding to the context and the label
        Each state will have the following transitions:
        1-step forward (expected) - 1 possible transition
        1-step back-slip - 1 possible transition
        1-step forward skip - 1 possible transition
        Self-loop
        '''
        # Create model and add states
        model = yahmm.Model(name=model_name) if not model_name is None \
                                                else yahmm.Model(name='HMM_linear_model')
        previous_skip = None
        previous_short_slip = None
        current_short_slip = None
        previous_mean = 0
        previous_sd = 0
        abasic = False # Abasic flag to trigger allowing re-read
        abasic_kmer = None # Track the position of abasic XXXX
        states_list = []

        for index in range(len(kmer_list)):

            # State name, mean and stdev for the kmer
            kmer = kmer_list[index]
            current_mean = kmer_current_dict[kmer][0]
            current_sd = kmer_current_dict[kmer][1]
            # Transition probabilities for a match state
            # Self-loop to itself
            self_loop = 0.001
            # End from any state, i.e. reach model.end
            end = 0.005
            # Transitions for Drop-off State
            drop = 0.001
            # Transitions for going to Blip State
            blip = 0.005
            blip_self = 0.1
            # Back Slips, short and long
            slip = 0.05 if index > 0 else 0.00
            # Only short backslips possible
            short_slip = slip
            long_slip = 0.0
            # Transitions from silent slip states
            # Short slip from silent short slip state
            step_back = 0.60
            # Skip that accounts for a missed segment
            skip = 0.2
            # Transitions from current skip silent state to the previous match state or
            # previous silent skip states
            long_skip = 0.05
            # Transitions for Insert state between two neighboring match states
            insert = 0.02 if index > 0 else 0.00
            # Self loop for an insert state
            ins_self = 0.05
            # Transition to the next match state (Forward Transition)
            # Each match state has transitions out to self_loop, end, drop, blip, slip,
            # skip, insert, re_read, and forward
            forward = 1 - (self_loop + end + blip + slip + skip + insert)

            # Create and Add State
            current_state = yahmm.State(yahmm.NormalDistribution(current_mean, \
                                        current_sd), \
                                        name = 'M_' + kmer + '_' + str(index))
            model.add_state(current_state)

            # Transitions for the match state
            # Self-loop to itself
            model.add_transition(current_state, current_state, self_loop)
            # The model could end from any match state
            if index < len(kmer_list) - 1:
                model.add_transition(current_state, model.end, end)

            # Each Match State can go to a silent drop-off state, and then to model.end
            drop_off = yahmm.State(None, name = 'S_DROPOFF_' + kmer + '_' + str(index))
            model.add_state(drop_off)
            # Transition to drop_off and back, from drop_off to end
            model.add_transition(current_state, drop_off, drop)
            model.add_transition(drop_off, current_state, 1.0 - blip_self)

            model.add_transition(drop_off, model.end, 1.00)

            # Each Match State can go to a Blip State that results from a voltage blip
            # Uniform Distribution with Mean and Variance for the whole event
            blip_state = yahmm.State(yahmm.UniformDistribution(15.0, 120.0), \
                                            name = 'I_BLIP_' + kmer + '_' + str(index))
            model.add_state(blip_state)
            # Self-loop for blip_staet
            model.add_transition(blip_state, blip_state, blip_self)
            # Transition to blip_state and back
            model.add_transition(current_state, blip_state, blip)
            model.add_transition(blip_state, current_state, 1.0 - blip_self)

            # Short Backslip - can go from 1 to the beginning but favors 1 > ...
            # Starts at state 1 when the first short slip silent state is created
            if index >= 1:
                # Create and add silent state for short slip
                current_short_slip = yahmm.State(None, name = 'B_BACK_SHORT_' + kmer + \
                                                            '_' + str(index))
                model.add_state(current_short_slip)
                # Transition from current state to silent short slip state
                model.add_transition(current_state, current_short_slip, short_slip)
                if index >= 2:
                    # Transition from current silent short slip state to previous
                    # match state
                    model.add_transition(current_short_slip, states_list[index-1], \
                                            step_back)
                    # Transition from current silent short slip state to previous silent
                    # short slip state
                    model.add_transition(current_short_slip, previous_short_slip, \
                                            1 - step_back)
                else:
                    model.add_transition(current_short_slip, states_list[index-1], 1.00)

            # Create and Add Skip Silent State
            current_skip = yahmm.State(None, name = 'S_SKIP_' + kmer + '_' + str(index))
            model.add_state(current_skip)

            if not previous_skip is None:
                # From previous Skip Silent State to the current Skip Silent State
                model.add_transition(previous_skip, current_skip, long_skip)
                # From previous Skip Silent State to the current match State
                model.add_transition(previous_skip, current_state, 1 - long_skip)

            # From previous match State to the current Skip Silent State
            if index == 0:
                model.add_transition(model.start, current_skip, 1.0 - forward)
            else:
                model.add_transition(states_list[index-1], current_skip, skip)

            # Insert States
            if index > 0:
                # Mean and SD for Insert State
                # Calculated as a mixture distribution
                insert_mean = (previous_mean + current_mean) / 2.0
                insert_sd =  numpy.sqrt(1/4 * ((previous_mean - current_mean) ** 2) \
                                        + 1/2 * (previous_sd ** 2 + current_sd ** 2))
                # Create and Add Insert State
                # Normal Distribution with Mean and Variance that represent
                # neighboring states
                insert_state = yahmm.State( yahmm.NormalDistribution(insert_mean, \
                                                insert_sd ), \
                                                name = 'I_INS_' + kmer + '_' + str(index))
                model.add_state(insert_state)
                # Self-loop
                model.add_transition(insert_state, insert_state, ins_self)
                # Transition from states_list[index-1]
                model.add_transition(states_list[index-1], insert_state, insert)
                # Transition to current_state
                model.add_transition(insert_state, current_state, 1.0 - ins_self)

            # Transition to the next match state
            if index == 0:
                # Only transitions from start to skip silent state or first match state
                model.add_transition(model.start, current_state, forward)
            elif index == 1:
                # Since I add match transitions from the previous match state to current
                # match state, I have to make sure the sum of outgoing edges adds to 1.0
                # For index 0, there is no slip, addition of M_0 -> M_1 happens at 1,
                # which means add this slip probability to the forward transition for M_0
                model.add_transition(states_list[index-1], current_state, forward + slip)
            else:
                model.add_transition(states_list[index-1], current_state, forward)

            # Append the current state to states list
            states_list.append(current_state)

            # Re-assign current states to previous states
            previous_skip = current_skip
            previous_short_slip = current_short_slip if not current_short_slip is None \
                                                        else None
            previous_mean = current_mean
            previous_sd = current_sd

            # Model end case
            if index == len(kmer_list) - 1:
                skip = 0.0
                insert = 0.0
                forward = 1 - (self_loop + end + blip + slip + skip + insert)
                # End cases
                model.add_transition(states_list[index], model.end, forward + end)
                model.add_transition(previous_skip, model.end, 1.00)

        model.bake()
        return model

    def HMM_means_model(self, model_name=None):
        '''
        This HMM models the basic structure of a MinION read
        '''
        # Create model and add states
        model = yahmm.Model(name=model_name) if not model_name is None \
                                                else yahmm.Model(name='HMM_read_model')

        abasic = False # Abasic flag to trigger allowing re-read
        abasic_kmer = None # Track the position of abasic XXXX

        state0 = State(NormalDistribution(70.2727, 1.9428), name='START')
        state1 = State(NormalDistribution(110.973, 2.2884), name='LEADER')
        state2 = State(MixtureDistribution([NormalDistribution(79.347, 2.8931),
                                           NormalDistribution(63.3126, 1.6572)],
                                           weights=[0.874, 0.126], frozen=True))
        state2.name = 'Adapter'
        state3 = State(NormalDistribution(113.5457, 2.4043), name='homopolymer')
        state4 = State(MixtureDistribution([NormalDistribution(79.679, 2.6393),
                                           NormalDistribution(105.784, 1.6572)],
                                           weights=[0.346, 0.654], frozen=True))
        state4.name = 'READ'

        #state5 = State(UniformDistribution(70.0, 140.0), name='Cliff')


        model.add_state(state0)
        model.add_state(state1)
        model.add_state(state2)
        model.add_state(state3)
        model.add_state(state4)
        #model.add_state(state5)

        #transitions from model.start
        model.add_transition(model.start, state0, 0.5) #to start
        model.add_transition(model.start, state1, 0.5) #to leader

        #transions from START state
        model.add_transition(state0, state0, 0.1) #self loop
        model.add_transition(state0, state1, 0.9) #to leader

        #transions from LEADER state
        model.add_transition(state1, state2, 0.1) #to adaapter
        model.add_transition(state1, state1, 0.9) #self loop

        #transions from Adapter state
        model.add_transition(state2, state3, 0.05) #to homopolymer
        model.add_transition(state2, state2, 0.95) #self loop

        #transions from HOMOPOLYMER state
        model.add_transition(state3, state4, 0.1) #to read
        model.add_transition(state3, state3, 0.9) #self loop
        #model.add_transition(state3, state5, 0.01) #to cliff

        #transions from Cliff state
        #model.add_transition(state4, state5, 0.01) #self loop
        #model.add_transition(state4, state3, 0.99) #to homopolymer

        #transions from READ state
        model.add_transition(state4, state4, 0.999) #self loop
        model.add_transition(state4, model.end, 0.001) #model end

        model.bake()
        return model

    def HMM_papi_model(self, model_name='pApI'):
        '''
        This HMM models the basic structure of a MinION read
        '''
        # Create model and add states
        model = yahmm.Model(name=model_name) if not model_name is None \
                                                else yahmm.Model(name='HMM_read_model')

        abasic = False # Abasic flag to trigger allowing re-read
        abasic_kmer = None # Track the position of abasic XXXX

        state0 = State(NormalDistribution(4.3025, 3.4873), name='POLYA')
        state1 = State(NormalDistribution(14.1195, 6.5471), name='POLYI')

        model.add_state(state0)
        model.add_state(state1)
        
        
        #transitions from model.start
        model.add_transition(model.start, state1, 1.0) #to polyI 

        #transions from polyI
        model.add_transition(state1, state1, 0.9) #self loop
        model.add_transition(state1, state0, 0.1) #to polyA

        #transions from polyA state
        model.add_transition(state0, state0, 0.95) #self loop
        model.add_transition(state0, model.end, 0.05) #to model end

        model.bake()
        return model

		
		
    def HMM_pa_model(self, model_name='pA'):
        '''
        This HMM models the basic structure of a MinION read
        '''
        # Create model and add states
        model = yahmm.Model(name=model_name) if not model_name is None \
                                                else yahmm.Model(name='HMM_read_model')


        state0 = State(NormalDistribution(4.3025, 3.4873), name='POLYA')

        model.add_state(state0)

        #transitions from model.start
        model.add_transition(model.start, state0, 1.0) #to polyA

        #transions from polyA
        model.add_transition(state0, state0, 0.964285714286) #self loop
        model.add_transition(state0, model.end, 0.0357142857143) #to model end

        model.bake()
        return model

		
    def HMM_pi_model(self, model_name='pI'):
        '''
        This HMM models the basic structure of a MinION read
        '''
        # Create model and add states
        model = yahmm.Model(name=model_name) if not model_name is None \
                                                else yahmm.Model(name='HMM_read_model')

        abasic = False # Abasic flag to trigger allowing re-read
        abasic_kmer = None # Track the position of abasic XXXX

        state0 = State(NormalDistribution(15.107535447472557, 6.5471), name='POLYI')

        model.add_state(state0)

        #transitions from model.start
        model.add_transition(model.start, state0, 1.0) #to polyI

        #transions from polyI
        model.add_transition(state0, state0, 0.976744186047) #self loop
        model.add_transition(state0, model.end, 0.023255813953) #to model end

        model.bake()
        return model
#############################################################################

##############################################################################
def model_maker(kmer_current_dict, model_name=None):
    kmer_list = map(str, range(len(kmer_current_dict)))
    model = HMM_Constructor().HMM_linear_model(kmer_list, kmer_current_dict, \
                                                model_name)
    return model
##############################################################################

##############################################################################
def prediction(models, sequences, algorithm = 'forward-backward'):

    # Predict sequence from HMM using a user-specified algorithm
    # Forward-Backward (default) or Viterbi
    sequence_from_hmm = []
    for i in range(len(sequences)):
        for model in models:
            if algorithm == 'viterbi':
                sequence_from_hmm.append(model.viterbi(sequences[i]))
            elif algorithm == 'forward-backward':
                sequence_from_hmm.append(model.forward_backward(sequences[i]))
    return sequence_from_hmm
##############################################################################

#############################################################################
def plot_event(filename, means, model):
    # Plots, top plot is segmented event colored in cycle by segments, bottom
    # subplot is segmented event aligned with HMM, colored by states
    plt.figure(figsize=(20, 8))
    plt.figure()
    plt.subplot(211)
    plt.grid()
    event.plot(color='cycle')
    plt.subplot(212)
    plt.grid()
    event.plot(color='hmm', hmm=model, cmap='Set1')
    fig_name = "{}_{}_{}.png".format(filename, event.start, event.end)
    #fig_name = filename + '_' + str(event.start) + '_' + str(event.end) + '.png'
    plt.tight_layout()
    plt.savefig(fig_name, format='png')
    plt.close()
    print "plotted figure {}".format(fig_name)
#############################################################################

#################################################################################3
def build_models(modelPath):
    '''
    Parses the sequence_summary.txt file from ONT base caller to map the
    file name to the strand name
    '''

    print >> sys.stderr, 'creating kmer current map'
    # CREATE CURRENT MAPS OUT OF MODELPATH
    # kmer_current_dict[index] = [meancurrent, stddev]
    # one dictionary per file where each entry is one line in the file
    models = []

    model_files = []
    for r, d, f in os.walk(modelPath):
        for thing in f:
            if '.txt' in thing:
                model_files.append(os.path.join(r,thing))


    for model_file in model_files:
        mName = model_file.split('/')[-1].split('.')[0]
        with open(model_file, 'r') as check:
            line = check.readline().strip()
            if line.startswith('0'):
                kmer_current_dict = kmer_current_map(model_file)
                model = model_maker(kmer_current_dict, model_name = mName)
                models.append(model)
            else:
                model = model_maker(kmer_current_dict, model_name = mName)
                models.append(model)
        '''
        Construct models: polyA, polyI, polyA_and_polyI, read_model
        '''
        # Build one model for each current_dict / filename in modelpath
        # model_maker takes kmer_list, which is just a list of str numbers that are the keys in that dict
        # then model_maker passes the list, dict, name into HMM_linear_model, which gets kmer (just an int),
        # and mean and stddev for each entry in the kmer_dict.
        # Make HMMs for every file and then add them all to a list called models.

    return models
#################################################################################

###############################################################################
def scale_pipe(fastqName, pathToFast5, outputDir, names, seq_sum = '', threads = 1):
    '''
    runs raw current parsing software
    '''

    events = []
    if os.path.isfile('{}.index'.format(fastqName)):
        pass
    else:
        if seq_sum:
            subprocess.call(['/projects/nanopore-working/bin/signal/binary/nanopolish',
                             'index', '-d', pathToFast5, '-s', seq_sum, fastqName])
        else:
            subprocess.call(['/projects/nanopore-working/bin/signal/binary/nanopolish',
                             'index', '-d', pathToFast5, fastqName])


    if any(File.endswith(".tsv") for File in os.listdir(outputDir)):
        pass
    else:
        subprocess.call(['/projects/nanopore-working/bin/signal/binary/nanopolish',
                         'dump-initial-alignment', '--reads', fastqName,
                         '--scale-events', '-o', outputDir, '-t', '{}'.format(threads)])

    for path, subdir, files in os.walk(outputDir):
        for file in files:
            if file.endswith(".tsv") and file.rsplit('.', 1)[0] in names:
                events.append(os.path.join(path, file))

    return events
###############################################################################

###############################################################################
def parse_tsv(infile):
    '''
    parses the tsv output from nanopolish scaling
    '''
    means = []
    stds = []
    duration = []

    with open(infile, 'r') as tsv:
        line = tsv.readline()
        for line in tsv:
            if line:
                line = line.strip().split('\t')
                means.append(float(line[3]))
                stds.append(float(line[4]))
                duration.append(float(line[6]))

    return np.array(means[::-1]), np.array(stds[::-1]), np.array(duration[::-1])
###############################################################################

###############################################################################
def plot_current(tsv, label):
    '''
    plots the raw current signal from the pipe
    '''

    means, stds, duration = parse_tsv(tsv)
    if means.any():
        fig_name = "{1}_{0}".format(tsv.rsplit('.',1)[0].rsplit('/',1)[-1], label)
        plt.figure()
        colormap = ['r', 'b', '#FF6600', 'g']
        color_cycle=[i%len(colormap) for i in xrange(len(means))]
        plt.plot(means) 
        plt.grid()
        plt.ylabel("mean current")
        plt.xlabel("time")
        plt.title(fig_name)
        plt.tick_params(which='both', bottom=True, top=False,
                        right=False, left=True)

        plt.savefig("{}.png".format(fig_name), dpi=300, format='png')
        plt.close()
################################################################################

#############################################################################
def make_var(stds):
    '''
    takes series of stdev and returns list of variances
    '''
    var = []

    for i in stds:
        var.append(i**2)

    return np.array(var)
#############################################################################

#############################################################################
def parse_path(inpath):
    '''
    parses the vitirbi path from the means alignment to find the indexes
    of the homopolymer segment for classification
    '''
    homopolymer = []
    with open(inpath, 'r') as path:
        for line in path:
            if not line.strip().startswith('mean'):
                line = line.strip().split('\t')
                if line[1] == 'homopolymer':
                    homopolymer.append(float(line[4]))

    return homopolymer
#############################################################################

#############################################################################
def calc_speed(pathname, fastq_length):
    '''
    
    '''

    read_duration = 0
    homo_duration = 0

    n = 0
    with open(pathname, 'r') as inpath:
        for line in inpath:
            if not line.startswith('mean'):
                #if n < 5:
                #    n += 1
                #    print line
                line = line.split('\t')
                if line[1].strip() == 'READ':
                    read_duration += float(line[5])

                elif line[1].strip() == 'homopolymer':
                    homo_duration += float(line[5])
    #print read_duration, homo_duration, fastq_length
    enzyme_speed = fastq_length / read_duration

    homopolymer_length = homo_duration * enzyme_speed

    return homopolymer_length, enzyme_speed
###############################################################################

###############################################################################
def separate_tail_lengths(pathname, enzyme_speed):
    '''
    estimates the tail length for the different tail types
    '''
    polyi_duration = 0
    polya_duration = 0

    n = 0
    with open(pathname, 'r') as inpath:
        for line in inpath:
            if not line.startswith('p'):
                #if n < 5:
                #    n += 1
                #    print line
                line = line.split('\t')
                if line[1].strip() == 'POLYA':
                    polya_duration += float(line[5])

                elif line[1].strip() == 'POLYI':
                    polyi_duration += float(line[5])

    return enzyme_speed * polyi_duration, enzyme_speed * polya_duration
#############################################################################

###############################################################################
def make_mean_and_variance_vs_time(pathname, dates):
    '''
    
    '''
    polya_means = []
    polya_vars = []

    polyi_means = []
    polyi_vars = []

    readname = os.path.basename(pathname).split('_')[0]
    print readname
    n = 0
    with open(pathname, 'r') as inpath:
        for line in inpath:
            if not line.startswith('p'):
                #if n < 5:
                #    n += 1
                #    print line
                line = line.split('\t')
                if line[1].strip() == 'POLYA':
                    polya_means.append(float(line[2]))
                    polya_vars.append(float(line[4]))

                elif line[1].strip() == 'POLYI':
                    polyi_means.append(float(line[2]))
                    polyi_vars.append(float(line[4]))
 
    if len(polya_means) > 0:
        polya_means = np.mean(polya_means)
    else:
        polya_means = -1

    if len(polya_means) > 0:
        polya_vars = np.mean(polya_vars)
    else:
        polya_vars = -1

    if len(polya_means) > 0:
        polyi_means = np.mean(polyi_means)
    else:
        polyi_means = -1

    if len(polya_means) > 0:
        polya_cars = np.mean(polyi_vars)
    else:
        polyi_vars = -1

    print dates[readname], polya_means, polya_vars, polyi_means, polyi_vars
    return dates[readname], polya_means, polya_vars, polyi_means, polyi_vars 
###############################################################################

#############################################################################
def model_pipe(events, label, fastq_lengths, dates, models = []):
    '''
    aligns the read currents to the models
    '''
    filecount = 0
    #If a model path is given, builds models from scratch
    if models:
        models = build_models(models)

        for model in models:
            print model
            with open("model_{}.txt".format(model.name), 'w') as writerFile:
                model.write(writerFile)
    #uses pre built models for each tail type
    else:
        #one model that records to the path of the read to classify
        #models = [HMM_Constructor().HMM_read_model('read_model')]
      
        #one model per tail type
        models = [HMM_Constructor().HMM_papi_model(),
                  HMM_Constructor().HMM_pa_model(),
                  HMM_Constructor().HMM_pi_model()]
        
        #model to find homopolymer
        means_models = [HMM_Constructor().HMM_means_model('means_model')]

    # Create blank templates for every model file
    viterbi_prediction = []
    # printing template
    column_output_template = '{0:>} {1:>5}'
    data_output_template = '{0:>d} {1:>5d}'

    counts = collections.defaultdict(int)
    for model in models:
        counts[model.name] = 0
    accuracy = 0.0

    matrix = {}
    for model in models:
        matrix[model.name] = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}

    fileCount = 0

    #Used for training
    training = []

    #classification dictionary
    tail_class = collections.defaultdict(int)
    index2name = collections.defaultdict()
    for n, model in enumerate(models):
        tail_class[model.name] = 0
        index2name[n] = model.name

    tsn = 0
    scores = []
    sequences = []
    fails = 0
    test_length = 0
    #sys.stdout.write("{}\t{}\t{}\t{}\t{}\t{}\n".format('label', 'read_ID', 'classification', 'homopolymer_length', 'poly(I) length', 'poly(A) length'))
    for file in events:
        if file:
            filecount += 1
            means, stds, duration = parse_tsv(file)

            if means.any():
                read_ID = os.path.split(file)[1].rsplit('.', 1)[0]
                sequences = [means]
                #sequences = [make_var(stds)]
                tsn += 1

                pred = prediction(means_models, sequences, algorithm = 'viterbi')
                '''
                sequenceSums += pred[0][0]
                sequenceCounts += 1
                sequenceScores.append(pred[0][0])
                scores = [float(pred[0][0])]
                '''
                
                pathname = "{}_means_path.txt".format(file.rsplit('.',1)[0])
                #print pathname
                try:
                    with open(pathname, 'w') as outfile:
                        for i, state in enumerate(pred[0][1]):
                            if '-start' in state[1].name:
                                outfile.write("{}\n".format(state[1].name))
                            elif '-end' in state[1].name:
                                outfile.write("{}\n".format(state[1].name))
                            else:
                                outfile.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(i-1, 
                                                                            state[1].name,
                                                                            means[i-1],
                                                                            stds[i-1],
                                                                            stds[i-1]**2,
                                                                            duration[i-1]))

                    sequences = [parse_path(pathname)[2:]]
                    
                    homopolymer_length, enzyme_speed = calc_speed(pathname, fastq_lengths[read_ID])
                    
                    training.append(sequences)
                    if len(sequences[0]) > 0:    
                        pred = prediction(models, sequences, algorithm = 'viterbi')
                        scores = []
                        for score in pred:
                            scores.append(float(score[0]))

                        #print pred[0][0], pred[1][0], pred[2][0]
                        #print scores.index(max(scores))
                        tail_class[index2name[scores.index(max(scores))]] += 1
                    
                        pathname = "{}_stds_path.txt".format(file.rsplit('.',1)[0])
                        #sys.stderr.write('{}\n'.format(scores.index(max(scores))))
                        #sys.stderr.write('{}\n'.format(len(pred)))#print pred[scores.index(max(scores))]
                    
                        with open(pathname, 'w') as outfile:
                            for i, state in enumerate(pred[scores.index(max(scores))][1]):
                                if '-start' in state[1].name:
                                    outfile.write("{}\n".format(state[1].name))
                                elif '-end' in state[1].name:
                                    outfile.write("{}\n".format(state[1].name))
                                else:
                                    outfile.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(i-1,
                                                                                    state[1].name,
                                                                                    means[i-1],
                                                                                    stds[i-1],
                                                                                    stds[i-1]**2,
                                                                                    duration[i-1]))
                        
                        polyi_length, polya_length = separate_tail_lengths(pathname, enzyme_speed)
                        #scores.append(float(model[0]))
                        #print scores
                        #classified_model = scores.index(max(scores))
                        #print classified_model

                        #sys.stdout.write("{}\t{}\t{}\t{}\n".format(label, read_ID, index2name[scores.index(max(scores))], homopolymer_length))
                        sys.stdout.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(label, read_ID, index2name[scores.index(max(scores))], homopolymer_length, polyi_length, polya_length))

                    else:
                        sys.stdout.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(label, read_ID, 'no match', -1, -1, -1))
                        #sys.stdout.write("{}\t{}\t{}\t{}\n".format(label, read_ID, 'no match', -1))
                except TypeError:
                    fails += 1
                    sys.stderr.write("{} failed\n".format(file))

    sys.stderr.write("{} fails\n".format(fails))
    '''
    n = 0
    if label == 'GLuc200i15':
        n = 2
    elif label == 'GLuc200A44':
        n = 1
    elif label == 'GLuc200A44i15':
        n = 0
    '''

    #print tsn
    #print filecount
    #print models[n].name
    #print label
    #with open("test_{}_model.txt".format(label), 'w') as outwrite:
    #    models[n].write(outwrite)
    #improvement = models[n].train(sequences, algorithm = 'viterbi')
    #print improvement
    #with open("test_{}_model_trained.txt".format(label), 'w') as trainwrite:
    #    models[n].write(trainwrite)

    #sys.stderr.write("Nanopolish passes: {}\n total tsv: {}\n\n".format(tsn, filecount))
    sys.stderr.write('{}\n'.format(label))
    for call in tail_class:
        sys.stderr.write("{}\t{}\t{:0.4f}\n".format(call, tail_class[call], float(tail_class[call]) / sum(tail_class.values())))
###############################################################################

###############################################################################
def count_fastq_lengths(fastq):
    '''
    makes name to length dictionary
    '''
    fastq_lengths = collections.defaultdict(int)

    for record in SeqIO.parse(fastq, "fastq":
        fastq_lengths[record.name] = len(record.seq)

    return fastq_lengths
###############################################################################        

###############################################################################
def random_sample(events, n = 0):
    '''
    randomly samples events    
    '''

    #comment out when not testing
    random.seed(1)

    if n >= 1.0:
        if n >= len(events):
            sys.stderr.write("\nError, random sample number {} greater than sample size {}\n\n".format(n, len(events)))
            subsample = events
        else:
            subsample = random.sample(events, int(n))

    elif n > 0.0 and n < 1.0:
        if n * len(events) >= len(events):
            sys.stderr.write("\nError, random sample number {} greater than sample size {}\n\n".format(n * len(events), len(events)))
            subsample = events
        else:
            subsample = random.sample(events, int(n * len(events)))

    else:
        subsample = events

    return subsample
###############################################################################

###############################################################################
def main(args):
    '''
    main function:
    base calls RNA fast5 files
    alings the reads
    and performs nanopolish eventalign
    '''
  
    sys.stderr.write("\n")
    start_time = time.time()
    options =  parse_args(args)

    if options.infile:
        with open(options.infile, 'r') as infile:
            sys.stdout.write("{}\t{}\t{}\t{}\n".format('label', 'read_ID', 'classification', 'homopolymer_length'))
            for line in infile:
                line = line.strip().split('\t')
                '''                
                fastq = line[0]
                label = line[1]
                path = line[2]
                outputDir = line[3]
                seq_sum = line[4]
                '''
                fastq_lengths = count_fastq_lengths(line[0])

                
                events = scale_pipe(line[0], line[2], line[3], fastq_lengths.keys(), 
                                    line[4], options.threads)

                subsample = random_sample(events, options.random_sample)

                model_pipe(subsample, line[1], fastq_lengths, options.models)
    else:
        sys.stdout.write("{}\t{}\t{}\t{}\n".format('label', 'read_ID', 'classification', 'homopolymer_length'))
        fastq_lengths = count_fastq_lengths(options.fastq)

        events = scale_pipe(options.fastq, options.pathToFast5, 
                            options.outputDir, fastq_lengths.keys(), 
                            options.sequencing_summary, options.threads)

        subsample = random_sample(events, options.random_sample)

        model_pipe(subsample, options.label, fastq_lengths, options.models)

    sys.stderr.write("took {} sec\n".format(time.time()-start_time))
###############################################################################

if (__name__ == "__main__"):
    main(sys.argv)
    raise SystemExit
