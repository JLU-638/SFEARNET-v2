python ../../../infer_ai2_sfearnet_test.py \
  --data_dir ../../../data/datasets/CLCD_256 \
  --checkpoint ../../../logs/CLCD_CD/SFEARNet_modify/lr-0.0001_bs-8_wd-0.001_lam-1.0_ep-100_seed-0/run_2026-07-24_08-01-40/best_model/best_model.pth \
  --prediction_dir ../../../results/ablation_predictions/CLCD/pseudo-on-bound-dhdm-on \
  --test_csv ../../../results/ablation_clcd/CLCD_pseudo-on-bound-dhdm-on_test.csv \
  --gpu_id 0