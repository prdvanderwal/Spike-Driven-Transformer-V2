#!/bin/bash

# Variables
SOURCE_DIR="/home4/p315895/Spike-Driven-Transformer-V2/data/cifar10"       # Replace with your source folder
DEST_DIR="/scratch/hb-prismaset/datasets"    # Replace with your destination directory
ARCHIVE_NAME="$(basename "$SOURCE_DIR").tar.gz"

mv ../group_DRO-master ../../../projects/p315895/
mv ../Negative-Attention ../../../projects/p315895/redundant/
