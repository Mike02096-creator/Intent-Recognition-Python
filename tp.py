import numpy as np
import glob
from sklearn import neighbors
from sklearn.metrics import confusion_matrix

f0_files=glob.glob('./BabyEars_data/*.f0')
en_files=glob.glob('./BabyEars_data/*.en')


f0_database=np.array([[]])
en_database=np.array([[]])
database_target=np.array([])

for f0_file in f0_files:
    f0_sample=np.loadtxt(f0_file)

    if 'ap' in f0_file:
        file_class='ap'
    elif 'pr' in f0_file:
        file_class='ap'
    elif 'at' in f0_file:
        file_class='at'

    if file_class=='ap' or file_class=='pr':
        local_derivative=(f0_sample[1:,1]-f0_sample[:-1,1])-(f0_sample[1:,0]-f0_sample[:-1,0])
        f0_wo_zeros_mask=f0_sample[:,1]!=0
        f0_sample=f0_sample[f0_wo_zeros_mask,:]

        database_target=np.concatenate((database_target,np.array([file_class])))

        mean_functional=np.mean(f0_sample[:,1])
        max_functional=np.max(f0_sample[:,1])
        range_functional=np.max(f0_sample[:,1])-np.min(f0_sample[:,1])
        variance=np.var(f0_sample[:,1])
        median=np.median(f0_sample[:,1])
        first_quartile=np.quantile(f0_sample[:,1],0.25)
        third_quartile=np.quantile(f0_sample[:,1],0.75)
        mean_absolute_local_derivative=np.mean(np.abs(local_derivative[f0_wo_zeros_mask[0:-1]]))
        if f0_database.shape[1]==0:
            f0_database=np.array([[mean_functional,max_functional,range_functional,variance,median,first_quartile,third_quartile,mean_absolute_local_derivative]])
        else:
            f0_database=np.concatenate((f0_database,np.array([[mean_functional,max_functional,range_functional,variance,median,first_quartile,third_quartile,mean_absolute_local_derivative]])),axis=0)
        
        en_file=f0_file[:-2]+'en'
        en_sample=np.loadtxt(en_file)

        local_derivative = (en_sample[1:,1]-en_sample[:-1,1])-(en_sample[1:,0]-en_sample[:-1,0])
        en_sample=en_sample[f0_wo_zeros_mask,:]

        mean_functional=np.mean(en_sample[:,1])
        max_functional=np.mean(en_sample[:,1])
        range_functional=np.max(en_sample[:,1])-np.min(en_sample[:,1])
        variance=np.var(en_sample[:,1])
        median=np.median(en_sample[:,1])
        first_quartile=np.quantile(en_sample[:,1],0.25)
        third_quartile=np.quantile(en_sample[:,1],0.75)
        mean_absolute_local_derivative=np.mean(np.abs(local_derivative[f0_wo_zeros_mask[0:-1]]))
        if en_database.shape[1]==0:
            en_database=np.array([[mean_functional,max_functional,range_functional,variance,median,first_quartile,third_quartile,mean_absolute_local_derivative]])
        else:
            en_database=np.concatenate((en_database,np.array([[mean_functional,max_functional,range_functional,variance,median,first_quartile,third_quartile,mean_absolute_local_derivative]])),axis=0)

baby_ears_database=np.concatenate((f0_database,en_database),axis=1)

# Build training database and a testing database

n_neighbors = 1

train_test_ratio = 0.3;

long = len(baby_ears_database);
test_samples = int(abs (train_test_ratio*long));
rand_idx = np.random.permutation(long);
train = rand_idx[:test_samples];
test = rand_idx[test_samples+1:];

#for (train_idx, test_idx) in enumerate(baby_ears_database):
training_database = baby_ears_database[train];
training_database_target = database_target[train];
testing_database = baby_ears_database[test];
testing_database_target = database_target[test];

model = neighbors.KNeighborsClassifier(n_neighbors)

model.fit(baby_ears_database, database_target)

yfit = model.predict(testing_database_target)

conf_matrix = confusion_matrix(yfit,testing_database_target);

#Accuracy (number of yfit==testing_database_targets / nb of samples (size of testing_database_targets)
#Accuracy = (conf_matrix[1,1]+conf_matrix[2,2])/ long(testing_database_target);
