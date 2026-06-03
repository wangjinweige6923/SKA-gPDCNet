# A Multi-Disease Classification Method for Fundus Images Integrating Gated Central Pixel-Difference Convolution and Semantic Knowledge Alignment

<img width="8943" height="4251" alt="Figure 1 SKA-gPDCNet Network Architecture_1" src="https://github.com/user-attachments/assets/40e49a75-af0b-43a5-83fa-4eb6f0daa57f" />

pip install -r requirements.txt

## Dataset and Pretrained Models

Due to GitHub storage limitations, datasets, pretrained models, and results are hosted externally.

Download (Data + Models + Results):

Datas：https://drive.google.com/file/d/1qzs5WUCh8af9l4Ken8s6r_obH2YpR3tg/view?usp=drive_link

https://drive.google.com/file/d/1eG7gZWJmymKwQ_YkAERVJrBrahZxUyXi/view?usp=drive_link

Models：

Results：https://drive.google.com/file/d/1zyGY7HhX3RvJmxSVmr_FH25y3QbBd1s5/view?usp=drive_link

python -m pip install -r requirements.txt

## Train

python .\run_paper_experiment_backbone.py --data-dir .\dataset --output-root .\runs\resnet34_5seed --seeds 20 40 42 80 100 --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --epochs 30 --batch-size 8 --img-size 224 224 --device cuda --num-workers 4 --grouping-mode class_base_id --backbone resnet34

python .\run_paper_experiment_backbone.py --data-dir .\dataset --output-root .\result\literature_adapted_comparison\methods\ours_ska_2026 --seeds 110 220 900 1100 1110 --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --epochs 30 --batch-size 8 --img-size 224 224 --device cuda --num-workers 0 --grouping-mode class_base_id --backbone resnet34_ska_gpdc_c_v15 --model-label Ours --representative-seed 220


python .\run_paper_experiment_backbone.py --data-dir .\dataset --output-root .\result\ska_gpdcnet_no_vessel\ska_gpdcnet_fixed5seed --seeds 880 220 20 900 660 --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --epochs 30 --batch-size 8 --img-size 224 224 --device cuda --num-workers 0 --grouping-mode class_base_id --backbone resnet34_ska_gpdc_c_v15 --model-label SKA-gPDCNet --representative-seed 220


## Evaluation

Examples:

python .\plot_baseline_paper_figures.py --runs-root .\runs --output-dir .\runs\baseline_visualization --seeds 20 40 42 80 100 

python .\run_literature_adapted_comparison.py --data-dir .\dataset --output-root .\result\literature_adapted_comparison --methods ours_ska_2026 --seeds 110 220 900 1100 1110 --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --epochs 30 --batch-size 8 --img-size 224 224 --device cuda --num-workers 0 --grouping-mode class_base_id --representative-seed 220

python .\run_ska_gpdc_experiment.py --data-dir .\dataset --output-root .\result --experiment-name ska_gpdcnet_no_vessel --seeds 880 220 20 900 660 --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --epochs 30 --batch-size 8 --img-size 224 224 --device cuda --num-workers 0 --grouping-mode class_base_id --representative-seed 220 --skip-existing

