python ../../../infer_ai2_sfearnet_test.py \
  --data_dir ../../../data/datasets/GZ-CD_256 \
  --checkpoint ../../../logs/GZ_CD/SFEARNet_modify/lr-0.0001_bs-8_wd-0.001_lam-1.0_ep-100_seed-0/run_2026-07-26_06-07-57/best_model/best_model.pth \
  --prediction_dir ../../../results/ablation_predictions/GZ/pseudo-off-dhdm-on \
  --test_csv ../../../results/ablation_gz/GZ_pseudo-off-dhdm-on_test.csv \
  --gpu_id 0