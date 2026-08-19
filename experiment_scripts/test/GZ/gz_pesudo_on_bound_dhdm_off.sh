python ../../../infer_ai2_sfearnet_test.py \
  --data_dir ../../../data/datasets/GZ-CD_256 \
  --checkpoint ../../../logs/GZ_CD/SFEARNet_modify/lr-0.0001_bs-8_wd-0.001_lam-1.0_ep-100_seed-0/run_2026-07-27_07-26-48/best_model/best_model.pth \
  --prediction_dir ../../../results/ablation_predictions/GZ/pseudo-on-bound-dhdm-off \
  --test_csv ../../../results/ablation_gz/GZ_pseudo-on-bound-dhdm-off_test.csv \
  --gpu_id 0