#!/bin/bash

name=$1
dir=$2
wget -O "${dir}/data/${name}.pdb" "https://bioinformatics.lt/ppi3d/download/interface_coordinates/${name}.pdb"
voronota-js-fast-iface-contacts --input "${dir}/data/${name}.pdb"  --coarse-grained --expand-ids > "${dir}/data/rp_${name}.tsv"
voronota-contacts --input "${dir}/data/${name}.pdb" --contacts-query ' --inter-residue  --no-solvent  --no-poly-bonds  --match-first c<B> --match-second c<B> --match-min-area 20  --match-min-seq-sep 0 ' --use-hbplus --tsv-output > "${dir}/data/rr_${name}.tsv"
