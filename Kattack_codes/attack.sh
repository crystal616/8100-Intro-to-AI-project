#attack decision tree model

# attack natural xgb model
nohup python xgbKantchelianAttack.py -d=../chosen_sample/xgb/breast_cancer_xgb_samples.svm -m=../models/xgb/breast_cancer_xgb.model -c=2 >../Kattack_result/xgb/breast_cancer_xgb_log.txt 2>&1 &
#covtype todo
nohup python xgbKantchelianAttack.py -d=../chosen_sample/xgb/covtype_xgb_samples.svm -m=../models/xgb/covtype_xgb.model -c=7 >../Kattack_result/xgb/covtype_xgb_log.txt 2>&1 &
#diabetes todo
nohup python xgbKantchelianAttack.py -d=../chosen_sample/xgb/diabetes_xgb_samples.pkl -m=../models/xgb/diabetes_xgb.model -c=2 >../Kattack_result/xgb/diabetes_xgb_log.txt 2>&1 &
#MNIST mulclass
nohup python xgbKantchelianAttack.py -d=../chosen_sample/xgb/MNIST2_6_xgb_samples.pkl -m=../models/xgb/MNIST2_6_xgb.model -c=2 >../Kattack_result/xgb/MNIST2_6_xgb_log.txt 2>&1 &
# attack robust xgb model
#nohup python xgbKantchelianAttack.py -d=data/ori_mnist.test0 -m=models/rxgb/robust_mnist_0200.model -c=10 >robust.log 2>&1 $
nohup python xgbKantchelianAttack.py -d=../chosen_sample/rxgb/breast_cancer_rxgb_samples.s -m=../models/rxgb/breast_cancer/breast_cancer_rxgb.model -c=2 >../Kattack_result/rxgb/breast_cancer_rxgb_log.txt 2>&1 &
nohup python xgbKantchelianAttack.py -d=../chosen_sample/rxgb/cod_rna_rxgb_samples.s -m=../models/rxgb/cod_rna/cod_rna_rxgb.model -c=2 >../Kattack_result/rxgb/cod_rna_rxgb_log.txt 2>&1 &
#covtype todo 5000
nohup python xgbKantchelianAttack.py -d=../chosen_sample/rxgb/covtype_rxgb_samples.s -m=../models/rxgb/covtype/covtype_rxgb.model -c=7 >../Kattack_result/rxgb/covtype_rxgb_log.txt 2>&1 &
nohup python xgbKantchelianAttack.py -d=../chosen_sample/rxgb/diabetes_rxgb_samples.s -m=../models/rxgb/diabetes/diabetes_rxgb.model -c=2 >../Kattack_result/rxgb/diabetes_rxgb_log.txt 2>&1 &
nohup python xgbKantchelianAttack.py -d=../chosen_sample/rxgb/MNIST2_6_rxgb_samples.s -m=../models/rxgb/MNIST2_6_rxgb.model -c=2 >../Kattack_result/rxgb/MNIST2_6_rxgb_log.txt 2>&1 &