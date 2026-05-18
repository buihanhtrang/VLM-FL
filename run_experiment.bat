@echo off
echo Starting PromptFL experiment on CIFAR-100...
call conda activate CITE
python federated_main.py ^
  --dataset-config-file configs/datasets/cifar100.yaml ^
  --config-file configs/trainers/PromptFolio/rn50.yaml ^
  --model fedavg ^
  --trainer PromptFL ^
  --num_users 5 ^
  --beta 0.3 ^
  --round 3 ^
  --root ./data ^
  --partition noniid-labeldir ^
  --train_batch_size 8 ^
  --test_batch_size 8 ^
  --useall True
echo Done! Results saved to result.json
pause
