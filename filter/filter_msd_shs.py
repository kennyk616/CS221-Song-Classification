import os
import sys
import json
import re
import time


def parse_clique(line):
    tokens = line[1:].split(',')
    return {'works':tokens[:-1], 'name':tokens[-1]}

def parse_track(line):
    tokens = line.split("<SEP>")
    return {'TID': tokens[0], 'AID': tokens[1], 'work': tokens[2]}


def tid_path_index(data_dir):
    index = {}

    def wcallback(index, dirname, fnames):
        # fd = {os.path.splitext(f)[0]:os.path.relpath(f, top) for f in fnames if os.path.isfile(f)}
        fd = {os.path.splitext(f)[0]:os.path.join(dirname,f) for f in fnames}
        index.update(fd)

    os.path.walk(data_dir, wcallback, index)
    return index

def construct_path(TID):
    letters = TID[2:5]
    return os.path.join(letters[0], letters[1], letters[2], TID + ".h5")

def main(args):

    # Index data directory
    # t0 = time.time()
    # print "Building path index..."
    # startdir = os.path.abspath(os.curdir)
    # os.chdir(args.datadir)
    # path_index = tid_path_index("./data/")
    # os.chdir(startdir)
    # print "Completed in %.02f s" % (time.time() - t0)


    # Cliques identified by name:(work1,work2,)
    cliques = dict()

    # Tracks identified by TID:(TID,AID,work, clique_name)
    tracks = dict()
    paths = dict()

    # Build dictionaries from shs file
    t0 = time.time()
    print "Parsing shs file..."

    counterTrack = 0
    counterClique = 0
    counterFound = 0
    with open(args.shsfile) as f:
        last_clique_name = ""
        for line in f:
            if line[0] == "#":
                continue
            elif line[0] == "%":
                c = parse_clique(line)
                cliques[c['name']] = c
                last_clique_name = c['name']
                counterClique += 1
            else:
                entry = parse_track(line)
                entry['clique_name'] = last_clique_name
                TID = entry['TID']
                counterTrack += 1

                # if TID in path_index:
                    # paths[TID] = path_index[TID]
                paths[TID] = construct_path(TID)
                tracks[TID] = entry
                counterFound += 1

            if args.count and counterTrack >= args.count: break

    print "%d tracks found in %d cliques." % (counterFound, counterClique)
    print "%d entries processed in %.02f s" % (counterTrack, time.time() - t0)


    # Output data
    if args.do_print:
        print "Cliques:"
        print cliques

        print "Tracks:"
        print tracks

        print "Track paths:"
        print paths

        print "Path index:"
        print path_index

    # Save dictionaries as .json
    outfile_name = os.path.splitext(args.shsfile)[0]
    os.mkdir(outfile_name)
    outfile_name = os.path.join(outfile_name, outfile_name)

    cliquefile = outfile_name + ".cliques.json"
    trackfile = outfile_name + ".tracks.json"
    pathfile = outfile_name + ".trackpaths.json"

    with open(cliquefile, 'w') as f:
        json.dump(cliques, f, indent=4)

    with open(trackfile, 'w') as f:
        json.dump(tracks, f, indent=4)

    with open(pathfile, 'w') as f:
        json.dump(paths, f, indent=4)


    # To feed to a shell script for file filtering
    pathlistfile = outfile_name + ".pathlist.txt"
    with open(pathlistfile, 'w') as pf:
        for entry in paths.values():
            pf.write(entry.strip("./") + "\n")


    # return cliques, tracks, paths, path_index


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--shs", dest='shsfile', required=True)
    parser.add_argument("-d", "--datadir", dest='datadir', required=True)
    parser.add_argument("-c", "--count", dest='count', type=int)

    parser.add_argument('-p', dest='do_print', default=False)

    args = parser.parse_args()
    main(args)
