This directory is concerned with the preprocessing of patches.  The tool
here produces a representation of data that can be passed to the tool found
in the deeplearning directory.  This tool is standalone because in practice
it may be useful to perform the preprocessing once, and then experiment
with the hyperparameters of the deep learning model to improve the results.

The following installation has been tested on Ubuntu 16.4 and 18.4 systems.

The preprocessing requires that the following software be installed:

OCaml: https://opam.ocaml.org/
Coccinelle: https://github.com/coccinelle/coccinelle
Parmap: https://github.com/rdicosmo/parmap

Parmap can also be installed using opam:

https://opam.ocaml.org/
https://opam.ocaml.org/packages/parmap/

The use of this code on Windows is discouraged, as parmap (and thus
parallelism) will not be available.

The preprocessor assumes the availability of python3 and the following
python3 libraries.

nltk: pip3 install nltk
enchant: pip3 install pyenchant

In python, it is necessary to run

>>> import nltk
>>> nltk.download('stopwords')

once before running the preprocessor.

---------------------------------

A typical command line is

   ./getinfo --commit-list <commit list> --git <git path> -o <prefix>

It is possible to specify --nolog to ignore the commit logs, --balance to
produce a balanced dataset by discarding some elements, and -j to indicate
the number of cores to use.  The default number of cores is 4.

This produces the files <prefix>.tmp (intermediate file), <prefix>.out
(representation of commits) and <prefix>.dict (dictionary for the commits).
Only <prefix>.out has to be provided to the deep learning process via the
--data command line argument.

For simplicity, the commit list requires labels for both training and
production data, but for production data the labels are ignored.

testdata in the current directory is a sample labelled list of commits.  These
commits are found in the Linux kernel, available at:

git://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git




-----------------------------
## Generate Data .pkl from .out files:
please modify the paths(input and output) and run text2dict.py to generate train.pkl and test.pkl.

      $ python text2dict.py -text_path  [path of text data] -dict_path [path of the dictionary data want to store]  -print True
   Example:
      
      $ python text2dict.py -text_path  'train_data.out' -dict_path 'train.pkl'  
      $ python text2dict.py -text_path  'test_data.out' -dict_path 'test.pkl' 
      
please modify the paths(input and output) and run generate_dict.py to generate dict.pkl.

      $ python generate_dict.py -text_path1 [path of our data1] -text_path2 [path of our data2] -dict_path [path we want to store dict.pkl]
   Example:
    
      $ python generate_dict.py -text_path1 'training_data.out' -text_path2 'test_data.out' -dict_path 'dict.pkl'
   Notes:
   training_data.out is the "text format" patches as training dataset (used in trainig phase).
   
   test_data.out is the "text format" patches as test dataset (used in evaluation phase).
   
   The reason why we need evaluation data (test_data.out) is that if we only build a dictionary based on training dataset (training_data.out), there may be some words in test_data.out which never appear in training_data.out. In this case, the generated "dict.pkl" is not the whole vacabulary. Considering it, I put both training data and test data to generate dict.pkl. As dict.pkl is consist of only token-id pairs, using test data will not affect the evaluation phase (no test info leak to model).
   
   If we don't want use test data in generating dict.pkl, we can change the command into this, to only use training data:
   
     $ python generate_dict.py -text_path1 'training_data.out' -text_path2 'training_data.out' -dict_path 'dict.pkl'

---------------------------------

All code found in this directory or any subdirectory is covered by GPLv2,
as described in license.txt.
The source code of PatchNet is available at
https://github.com/hvdthong/PatchNetTool
