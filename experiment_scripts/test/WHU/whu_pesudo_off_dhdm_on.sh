python ../../../infer_ai2_sfearnet_test.py \
  --data_dir ../../../data/datasets/WHU-CD-256 \
  --checkpoint ../../../logs/WHU_CD/SFEARNet_modify/lr-0.0001_bs-8_wd-0.001_lam-1.0_ep-100_seed-0/run_2026-07-25_08-46-39/best_model/best_model.pth \
  --prediction_dir ../../../results/ablation_predictions/WHU/pseudo-off-dhdm-on \
  --test_csv ../../../results/ablation_whu/WHU_pseudo-off-dhdm-on_test.csv \
  --gpu_id 4