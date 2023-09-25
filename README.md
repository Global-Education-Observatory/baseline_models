#### Baseline GEO Model training instructions

1. Login to the HPC using ```<WM_USERNAME>@vortex.sciclone.wm.edu```
2. If this is your first time working on the project, make a new directory using: ```mkdir geo```
3. Switch into the geo directory using ```cd geo```
4. If this is your first time working on the project, pull the code from https://github.com/Global-Education-Observatory/baseline_models.git using:
```
git init .
git remote add origin https://github.com/Global-Education-Observatory/baseline_models.git
git pull origin master
```
5. Make a new folder for all of your model runs using: ```mkdir models```
6. Run: 
```
source "/usr/local/anaconda3-2021.05/etc/profile.d/conda.csh"
module load anaconda3/2021.05
module load openmpi/3.1.4/gcc-9.3.0
unsetenv PYTHONPATH

conda create -n geo
```
7. Then, pip install each of the pacakges in packages.txt
8. Open train.py and change the username variable in line 28 to your W&M username
9. To train the model, run: ```qsub job```
