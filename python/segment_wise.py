#!/usr/bin/env python

from sys import stdout, stderr, exit
from os.path import isdir, join, basename
from os import makedirs
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter as ADHF
import csv


def readSSMs(data):
    """ ssms file reader """
    res = list()
    # segment table format
    # <chromosome id> <position> <variant count> <reference count>
    #
    for line in csv.reader(data, delimiter='\t'):
        res.append((line[0], int(line[1]), int(line[2]), int(line[3])))
    return res

def readSegments(data):
    """ segment file reader """
    res = list()
    # segment table format
    # <chromosome id> <start> <stop> <CN_a> <STD(CN_a)> <CN_b> <STD(CN_b)>
    #
    for line in csv.reader(data, delimiter='\t'):
        res.append((line[0], int(line[1]), int(line[2]), float(line[3]),
            float(line[4]), float(line[5]), float(line[6])))
    return res


def splitAndWrite(segments, ssms, out_dir, seg_out_prefix, ssms_out_prefix):
    """ split segments and ssms into independent pieces and write them to files
    in <out_dir> with prefices seg_out_prefix and ssms_out_prefix, respectively
    """

    j = 0
    for i in xrange(len(segments)):
        chr_id, start, end = segments[i][:3]
        # write current segment
        seg_out = open(join(out_dir, '%s_segid_%s' %(seg_out_prefix, i)), 'w')
        print >> seg_out, '\t'.join(map(str, segments[i]))
        seg_out.close()

        # skip SSMs that may lie before the current segment
        while j < len(ssms) and ssms[j][:2] < (chr_id, start):
            j += 1
        # write out SSMs associated with current segment
        ssms_out = open(join(out_dir, '%s_segid_%s' %(ssms_out_prefix, i)), 'w')
        while j < len(ssms) and ssms[j][:2] <= (chr_id, end):
            print >> ssms_out, '\t'.join(map(str, ssms[j]))
            j += 1
        ssms_out.close()
            

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class = ADHF)
    parser.add_argument('-o', '--output_dir', type=str, default='.',
            help='output directory')
#    parser.add_argument('-m', '--merge_single_copy', action='store_true',
#            help='move all SSMs that are located on a segment not affected ' + \
#                    'by a CNV onto one segment')
    parser.add_argument('segments_file', type=str,
            help='file containing CNV information')
    parser.add_argument('ssms_file', type=str,
            help='file containing SSMs information')
    args = parser.parse_args()
    
    if not args.output_dir:
        print >> stderr, 'ERROR: option for output directory cannot be empty'
        exit(1)
    elif not isdir(args.output_dir):
        makedirs(args.output_dir)
    
    segments = readSegments(open(args.segments_file))
    ssms = readSSMs(open(args.ssms_file))
    splitAndWrite(segments, ssms, args.output_dir, basename(args.segments_file),
            basename(args.ssms_file))

