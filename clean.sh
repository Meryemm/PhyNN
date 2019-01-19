echo 'Cleaning out all the temporary files: fasta, trees, nj, np'
echo 'keras, and results will be kept' 

mv data/fasta/.gitkeep data/.gitkeep

rm -rf data/fasta
mkdir data/fasta
cp data/.gitkeep data/fasta/.gitkeep

rm -rf data/trees
mkdir data/trees
cp data/.gitkeep data/trees/.gitkeep

rm -rf data/nj
mkdir data/nj
cp data/.gitkeep data/nj/.gitkeep

rm -rf data/np
mkdir data/np
cp data/.gitkeep data/np/.gitkeep

rm data/.gitkeep

rm -rf tf_logs
