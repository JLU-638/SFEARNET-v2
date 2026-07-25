python ../../../infer_ai2_sfearnet_test.py \
  --data_dir ../../../data/datasets/LEVIR_CD_256 \
  --checkpoint ../../../logs/LEVIR_CD/SFEARNet_modify/lr-0.0001_bs-8_wd-0.001_lam-1.0_ep-100_seed-0/run_2026-07-25_09-09-46/best_model/best_model.pth \
  --prediction_dir ../../../results/ablation_predictions/LEVIR/pseudo-on-dhdm-off \
  --test_csv ../../../results/ablation_levir/LEVIR_pseudo-on-dhdm-off_test.csv \
  --gpu_id 0