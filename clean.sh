echo 'Cleaning out all the temporary files: fasta, trees, nj, np'
echo 'keras, and results will be kept' 
rm -rf data/fasta
mkdir data/fasta
rm -rf data/trees
mkdir data/trees
rm -rf data/nj
mkdir data/nj
rm -rf data/np
mkdir data/np
rm -rf tf_logs
