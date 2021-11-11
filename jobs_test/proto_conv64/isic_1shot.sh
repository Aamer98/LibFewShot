#!/bin/bash
#SBATCH --mail-user=ar.aamer@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=proto_conv64_ISIC_1shot
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=127000M
#SBATCH --time=0-12:00
#SBATCH --account=rrg-ebrahimi

nvidia-smi

module load python/3.7
source ~/my_env6/bin/activate

echo "------------------------------------< Data preparation>----------------------------------"
echo "Copying the source code"
date +"%T"
cd $SLURM_TMPDIR
cp -r ~/scratch/LibFewShot .

echo "Copying the datasets"
date +"%T"
cp -r ~/scratch/LibFewShot_Dataset/* .

echo "Extract to dataset folder"
date +"%T"
cd LibFewShot/dataset

#tar -xf $SLURM_TMPDIR/CIFAR100.tar.gz
#tar -xf $SLURM_TMPDIR/CUB_200_2011_FewShot.tar.gz
#tar -xf $SLURM_TMPDIR/CUB_birds_2010.tar.gz
#tar -xf $SLURM_TMPDIR/StanfordCar.tar.gz
#tar -xf $SLURM_TMPDIR/StanfordDog.tar.gz
unzip $SLURM_TMPDIR/ISIC_lfs.zip

#tar -xf $SLURM_TMPDIR/miniImageNet--ravi.tar.gz
#cat $SLURM_TMPDIR/tieredImageNet.tar.gz* | tar -zxf -

echo "----------------------------------< End of data preparation>--------------------------------"
date +"%T"
echo "--------------------------------------------------------------------------------------------"

echo "---------------------------------------<Run the program>------------------------------------"
date +"%T"

cd ..
python run_test.py --data_root ./dataset/ISIC



#python run_test.py --data_root ./dataset/ISIC --weight-root ./results/ProtoNet-miniImageNet--ravi-Conv64F-5-5-Nov-10-2021-14-47-48


wait

cd $SLURM_TMPDIR
cp -r $SLURM_TMPDIR/LibFewShot/results/ ~/scratch/LibFewShot/