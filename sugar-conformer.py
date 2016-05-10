#!/usr/bin/env python3
import sys
import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import MDAnalysis as mda


def main(file_tuple):
    u = mda.Universe(*file_tuple)

    # Rename atoms
    u.atoms.set_types(["C", "C", "C", "C", "C", "O"])
    u.atoms.set_names(["C1", "C2", "C3", "C4", "C5", "O5"])
    u.atoms.guess_bonds()

    # Create dihedrals
    dih_selections = [u.atoms.C1 + u.atoms.C2 + u.atoms.C3 + u.atoms.C4,
                      u.atoms.C2 + u.atoms.C3 + u.atoms.C4 + u.atoms.C5,
                      u.atoms.C3 + u.atoms.C4 + u.atoms.C5 + u.atoms.O5,
                      u.atoms.C4 + u.atoms.C5 + u.atoms.O5 + u.atoms.C1,
                      u.atoms.C5 + u.atoms.O5 + u.atoms.C1 + u.atoms.C2,
                      u.atoms.O5 + u.atoms.C1 + u.atoms.C2 + u.atoms.C3]
    dihedrals = list(map(mda.core.topologyobjects.Dihedral, dih_selections))

    # Project onto canonical conformations
    ref_vectors = np.array([[60, 0, 60],
                        [-60, 60, -30],
                        [60, -60, -30],
                        [-60, 0, 60],
                        [60, 60, -30],
                        [-60, -60, -30]])
    ref_norms =np.linalg.norm(ref_vectors, axis=0)
    xs, ys, zs = [], [], []
    for ts in u.trajectory:
        dih_vector = np.fromiter(map(lambda x: x.dihedral(), dihedrals), dtype=np.float, count=6)
        contrib = np.dot(dih_vector, ref_vectors) / (ref_norms * np.linalg.norm(dih_vector))
        xs.append(contrib[0])
        ys.append(contrib[1])
        zs.append(contrib[2])

    # Classify on axis of 4C1 to 1C4
    confs = [sum(map(lambda x: x < -1/3, xs)) * 100 / len(xs),
             sum(map(lambda x: -1/3 <= x <= 1/3, xs)) * 100 / len(xs),
             sum(map(lambda x: x > 1/3, xs)) * 100 / len(xs)]
    print("4C1:     {0:4.2f} %".format(confs[0]))
    print("neutral: {0:4.2f} %".format(confs[1]))
    print("1C4:     {0:4.2f} %".format(confs[2]))


if __name__ == "__main__":
    if not len(sys.argv) in {2, 3}:
        print("Must provide trajectory filename(s), exiting.")
        sys.exit(-1)
    files = tuple(sys.argv[1:])
    main(files)
