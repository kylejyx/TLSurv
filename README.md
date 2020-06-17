# TTSurv

Paper link: [Link](doi.org)

For questions or comments, please feel free to contact jiang@u.nus.edu. 

## Instructions for using TTSurv(MAE)

Please replace "<env>" with the actual environment name.  
Please replace "<mod>" with the actual modality name such as "cnv mirna" or "mrna None".  
Please replace "[list of hyper-parameters]" with relevant hyper-parameters or list of candidate hyper-parameters such as "25 [0.1,0.2] 64". Please ensure there is no space in any hyper-parameter.   

Step 1. Prepare the conda environment by "conda create --name <env> --file spec-file.txt"  
Step 2. Activate the conda environment by "conda activate <env>"  
Step 3. Train the embedding network using "python Surv_Coxnnet.py <mod> [list of hyper-parameters]"  
Step 4. Collect results from log file folder by "python collect_results.py Coxnnet"  
Step 5. Rename the relevant state files and dict files for second stage  
Step 6. Train the fusion network using "python Surv_train_MAE.py <mod> [list of hyper-parameters]"  
Step 7. Collect results from log file folder by "python collect_results.py MAE"  
Step 8. Rename the relevant state files and dict files for third stage  
Step 9. Train the survival network using "Surv_MAESurv.py <mod> [list of hyper-parameters]"  
Step 10. Collect results from log file folder by "python collect_results.py MAESurv"  

