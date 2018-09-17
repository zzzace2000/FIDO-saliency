#!/usr/bin/env bash

mkdir result

#echo "Start doing vbd deeplift...."
#python deeplift_compare.py --model vbd --window 1 --from-digit 8 --to-digit 3 --verbose 0 --top_n -1 --no-cuda False --visualize False
#python deeplift_compare.py --model vbd --window 1 --from-digit 8 --to-digit 6 --verbose 0 --top_n -1 --no-cuda False --visualize False
#python deeplift_compare.py --model vbd --window 1 --from-digit 9 --to-digit 1 --verbose 0 --top_n -1 --no-cuda False --visualize False
#python deeplift_compare.py --model vbd --window 1 --from-digit 4 --to-digit 1 --verbose 0 --top_n -1 --no-cuda False --visualize False
#
#echo "Start doing p_b deeplift...."
#python deeplift_compare.py --model p_b --window 1 --from-digit 8 --to-digit 3 --verbose 0 --top_n -1 --no-cuda False --visualize False
#python deeplift_compare.py --model p_b --window 1 --from-digit 8 --to-digit 6 --verbose 0 --top_n -1 --no-cuda False --visualize False
#python deeplift_compare.py --model p_b --window 1 --from-digit 9 --to-digit 1 --verbose 0 --top_n -1 --no-cuda False --visualize False
#python deeplift_compare.py --model p_b --window 1 --from-digit 4 --to-digit 1 --verbose 0 --top_n -1 --no-cuda False --visualize False
#
#echo "Start doing p_b 2 deeplift...."
#python deeplift_compare.py --model p_b --window 2 --from-digit 8 --to-digit 3 --verbose 0 --top_n -1 --no-cuda False --visualize False
#python deeplift_compare.py --model p_b --window 2 --from-digit 8 --to-digit 6 --verbose 0 --top_n -1 --no-cuda False --visualize False
#python deeplift_compare.py --model p_b --window 2 --from-digit 9 --to-digit 1 --verbose 0 --top_n -1 --no-cuda False --visualize False
#python deeplift_compare.py --model p_b --window 2 --from-digit 4 --to-digit 1 --verbose 0 --top_n -1 --no-cuda False --visualize False
#
#echo "Start doing p_b 3 deeplift...."
#python deeplift_compare.py --model p_b --window 3 --from-digit 8 --to-digit 3 --verbose 0 --top_n -1 --no-cuda False --visualize False
#python deeplift_compare.py --model p_b --window 3 --from-digit 8 --to-digit 6 --verbose 0 --top_n -1 --no-cuda False --visualize False
#python deeplift_compare.py --model p_b --window 3 --from-digit 9 --to-digit 1 --verbose 0 --top_n -1 --no-cuda False --visualize False
#python deeplift_compare.py --model p_b --window 3 --from-digit 4 --to-digit 1 --verbose 0 --top_n -1 --no-cuda False --visualize False

#echo "Start doing vbd window 2...."
#python deeplift_compare.py --model vbd_window --window 2 --from-digit 8 --to-digit 3 --verbose 0 --top_n -1 --no-cuda False --visualize False
#python deeplift_compare.py --model vbd_window --window 2 --from-digit 8 --to-digit 6 --verbose 0 --top_n -1 --no-cuda False --visualize False
#python deeplift_compare.py --model vbd_window --window 2 --from-digit 9 --to-digit 1 --verbose 0 --top_n -1 --no-cuda False --visualize False
#python deeplift_compare.py --model vbd_window --window 2 --from-digit 4 --to-digit 1 --verbose 0 --top_n -1 --no-cuda False --visualize False
#
#echo "Start doing vbd window 3...."
#python deeplift_compare.py --model vbd_window --window 3 --from-digit 8 --to-digit 3 --verbose 0 --top_n -1 --no-cuda False --visualize False
#python deeplift_compare.py --model vbd_window --window 3 --from-digit 8 --to-digit 6 --verbose 0 --top_n -1 --no-cuda False --visualize False
#python deeplift_compare.py --model vbd_window --window 3 --from-digit 9 --to-digit 1 --verbose 0 --top_n -1 --no-cuda False --visualize False
#python deeplift_compare.py --model vbd_window --window 3 --from-digit 4 --to-digit 1 --verbose 0 --top_n -1 --no-cuda False --visualize False

# 1005 mnist vbdl1 cluster10
python mnist_compare_penalty.py --importance-method vbdl1 --prior 0.5  --reg-coef 1E-4 \
--num-imgs 20 --save-dir ./result/1005-vbdl1-1E-4/ --the-digit 8 \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10
python mnist_compare_penalty.py --importance-method vbdl1 --prior 0.5  --reg-coef 1E-3 \
--num-imgs 20 --save-dir ./result/1005-vbdl1-1E-3/ --the-digit 8 \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10
python mnist_compare_penalty.py --importance-method vbdl1 --prior 0.5  --reg-coef 1E-5 \
--num-imgs 20 --save-dir ./result/1005-vbdl1-1E-5/ --the-digit 8 \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10

# cluster 9
python mnist_compare_penalty.py --importance-method vbd --prior 0.5  --reg-coef 0.01 \
--num-imgs 20 --save-dir ./result/1005-vbd-p0.5-0.01/ --the-digit 8 \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10
python mnist_compare_penalty.py --importance-method vbd --prior 0.5  --reg-coef 0.001 \
--num-imgs 20 --save-dir ./result/1005-vbd-p0.5-0.001/ --the-digit 8 \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10
python mnist_compare_penalty.py --importance-method vbd --prior 0.5  --reg-coef 0.1 \
--num-imgs 20 --save-dir ./result/1005-vbd-p0.5-0.1/ --the-digit 8 \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10

# CLuster 8
python mnist_compare_penalty.py --importance-method vbd --prior 0.999  --reg-coef 1E-5 \
--num-imgs 20 --save-dir ./result/1005-vbd-p0.999-1E-5/ --the-digit 8 \
--batch-size 64 --lr 0.01 --ard_init 1. --epoch-print 10
python mnist_compare_penalty.py --importance-method vbd --prior 0.999  --reg-coef 1E-6 \
--num-imgs 20 --save-dir ./result/1005-vbd-p0.999-1E-6/ --the-digit 8 \
--batch-size 64 --lr 0.01 --ard_init 1. --epoch-print 10
python mnist_compare_penalty.py --importance-method vbd --prior 0.999  --reg-coef 1E-4 \
--num-imgs 20 --save-dir ./result/1005-vbd-p0.999-1E-4/ --the-digit 8 \
--batch-size 64 --lr 0.01 --ard_init 1. --epoch-print 10

# Run flip two in this case. cluster8
python mnist_compare_penalty.py --importance-method vbd_l1_opposite --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 1E-4 --epochs 1000 \
--num-imgs 100 --save-dir ./result/1007-vbd_l1_opposite-1E-4/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10

python mnist_compare_penalty.py --importance-method vbd_l1_opposite --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 1E-3 --epochs 1000 \
--num-imgs 100 --save-dir ./result/1007-vbd_l1_opposite-1E-3/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10

python mnist_compare_penalty.py --importance-method vbd_l1_opposite --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 1E-2 --epochs 1000 \
--num-imgs 100 --save-dir ./result/1007-vbd_l1_opposite-1E-2/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10

python mnist_compare_penalty.py --importance-method vbd_l1_opposite --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 0.1 --epochs 1000 \
--num-imgs 100 --save-dir ./result/1007-vbd_l1_opposite-0.1/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10

python mnist_compare_penalty.py --importance-method p_b --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 0.1 --epochs 1000 \
--num-imgs 100 --save-dir ./result/1007-p_b/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10

# Everything wrong! Cluster8
python mnist_compare_penalty.py --importance-method vbd_l1_opposite --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 1E-3 --epochs 600 \
--num-imgs 100 --save-dir ./result/1013-vbd_l1_opposite-1E-3/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10

python mnist_compare_penalty.py --importance-method vbd_l1_opposite --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 1E-4 --epochs 600 \
--num-imgs 100 --save-dir ./result/1013-vbd_l1_opposite-1E-4/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10

python mnist_compare_penalty.py --importance-method vbd_l1_opposite --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 1E-5 --epochs 600 \
--num-imgs 100 --save-dir ./result/1013-vbd_l1_opposite-1E-5/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10

# Cluster7
python mnist_compare_penalty.py --importance-method vbd_l1_opposite --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 1E-6 --epochs 600 \
--num-imgs 100 --save-dir ./result/1013-vbd_l1_opposite-1E-6/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10

python mnist_compare_penalty.py --importance-method p_b --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 0.1 --epochs 600 \
--num-imgs 100 --save-dir ./result/1013-p_b/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10

# Cluster8: 0.5
python mnist_compare_penalty.py --importance-method vbd_opposite --prior 0.5 \
--mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 0.1 --epochs 600 \
--num-imgs 100 --save-dir ./result/1013-vbd_opposite-0.5-0.1/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10 && python mnist_compare_penalty.py --importance-method vbd_opposite --prior 0.5 \
--mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 1.0 --epochs 600 \
--num-imgs 100 --save-dir ./result/1013-vbd_opposite-0.5-1.0/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10 && python mnist_compare_penalty.py --importance-method vbd_opposite --prior 0.5 \
--mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 0.01 --epochs 600 \
--num-imgs 100 --save-dir ./result/1013-vbd_opposite-0.5-0.01/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10

# Cluster 8: 0.
python mnist_compare_penalty.py --importance-method vbd_l1_opposite --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 0 --epochs 600 \
--num-imgs 100 --save-dir ./result/1013-vbd_l1_opposite-0/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10

# Train gen model
python train_gen_model.py

# Cluster 8: 0. with preservation game.
python mnist_compare_penalty.py --importance-method vbdl1 --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 0 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-vbd_l1-0/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10

# Cluster 8: BBMP SSR => Run a few examples
python mnist_compare_penalty.py --importance-method vbdl1 --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 0 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-vbd_l1-0/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10

# Run BBMP SSR for hyperparameter search cluster8
python mnist_compare_penalty.py --importance-method bbmp_ssr --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 0 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-bbmp-ssr-0/  \
--batch-size 64 --lr 0.005 --ard_init 1. --epoch-print 10

python mnist_compare_penalty.py --importance-method bbmp_ssr --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 1 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-bbmp-ssr-1/  \
--batch-size 64 --lr 0.005 --ard_init 1. --epoch-print 10

python mnist_compare_penalty.py --importance-method bbmp_ssr --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 0.1 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-bbmp-ssr-0.1/  \
--batch-size 64 --lr 0.005 --ard_init 1. --epoch-print 10

python mnist_compare_penalty.py --importance-method bbmp_ssr --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 0.01 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-bbmp-ssr-0.01/  \
--batch-size 64 --lr 0.005 --ard_init 1. --epoch-print 10

python mnist_compare_penalty.py --importance-method bbmp_ssr --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 1E-3 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-bbmp-ssr-1E-3/  \
--batch-size 64 --lr 0.005 --ard_init 1. --epoch-print 10

# Same thing. Do with BBMP-SDR cluster6
python mnist_compare_penalty.py --importance-method bbmp_sdr --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 0 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-bbmp-sdr-0/  \
--batch-size 64 --lr 0.005 --ard_init 1. --epoch-print 10

python mnist_compare_penalty.py --importance-method bbmp_sdr --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 0.1 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-bbmp-sdr-0.1/  \
--batch-size 64 --lr 0.005 --ard_init 1. --epoch-print 10

python mnist_compare_penalty.py --importance-method bbmp_sdr --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 0.01 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-bbmp-sdr-0.01/  \
--batch-size 64 --lr 0.005 --ard_init 1. --epoch-print 10

python mnist_compare_penalty.py --importance-method bbmp_sdr --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 1 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-bbmp-sdr-1/  \
--batch-size 64 --lr 0.005 --ard_init 1. --epoch-print 10

# Do with flip 2 and vbdl1 => cluster0
python mnist_compare_penalty.py --importance-method vbdl1 --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 1E-3 --epochs 600 \
--num-imgs 20 --save-dir ./result/1018-vbd_l1-1E-3/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10 && python mnist_compare_penalty.py --importance-method vbdl1 --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 1E-4 --epochs 600 \
--num-imgs 20 --save-dir ./result/1018-vbd_l1-1E-4/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10 && python mnist_compare_penalty.py --importance-method vbdl1 --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 1E-5 --epochs 600 \
--num-imgs 20 --save-dir ./result/1018-vbd_l1-1E-5/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10

# cluster 0
python mnist_compare_penalty.py --importance-method vbdl1 --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 1E-5 --epochs 600 \
--num-imgs 80 --save-dir ./result/1018-vbd_l1-1E-5/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10 --image-offset 20

# cluster 8
python mnist_compare_penalty.py --importance-method vbdl1 --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 1E-4 --epochs 600 \
--num-imgs 80 --save-dir ./result/1018-vbd_l1-1E-4/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10 --image-offset 20

python mnist_compare_penalty.py --importance-method vbdl1 --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 1E-3 --epochs 600 \
--num-imgs 80 --save-dir ./result/1018-vbd_l1-1E-3/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10 --image-offset 20

python mnist_compare_penalty.py --importance-method vbdl1 --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 1E-2 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-vbd_l1-1E-2/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10 --image-offset 0

python mnist_compare_penalty.py --importance-method vbdl1 --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 0.1 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-vbd_l1-0.1/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10 --image-offset 0


# Do with one flip :cluster0
python mnist_compare_penalty.py --importance-method bbmp_sdr --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 0 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-8-bbmp-sdr-0/  \
--batch-size 64 --lr 0.005 --ard_init 1. --epoch-print 10

python mnist_compare_penalty.py --importance-method bbmp_sdr --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 0.1 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-8-bbmp-sdr-0.1/  \
--batch-size 64 --lr 0.005 --ard_init 1. --epoch-print 10

python mnist_compare_penalty.py --importance-method bbmp_sdr --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 0.01 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-8-bbmp-sdr-0.01/  \
--batch-size 64 --lr 0.005 --ard_init 1. --epoch-print 10


python mnist_compare_penalty.py --importance-method bbmp_sdr --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 0.01 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-8-bbmp-sdr-0.01/  \
--batch-size 64 --lr 0.005 --ard_init 1. --epoch-print 10

python mnist_compare_penalty.py --importance-method bbmp_sdr --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 0.01 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-8-bbmp-sdr-0.01/  \
--batch-size 64 --lr 0.005 --ard_init 1. --epoch-print 10

# The rest:
# - bbmp-ssr => flip1: cluster9
python mnist_compare_penalty.py --importance-method bbmp_ssr --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 0 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-8-bbmp-ssr-0/  \
--batch-size 64 --lr 0.005 --ard_init 1. --epoch-print 10

python mnist_compare_penalty.py --importance-method bbmp_ssr --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 0.01 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-8-bbmp-ssr-0.01/  \
--batch-size 64 --lr 0.005 --ard_init 1. --epoch-print 10

python mnist_compare_penalty.py --importance-method bbmp_ssr --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 0.1 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-8-bbmp-ssr-0.1/  \
--batch-size 64 --lr 0.005 --ard_init 1. --epoch-print 10

python mnist_compare_penalty.py --importance-method bbmp_ssr --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 1E-3 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-8-bbmp-ssr-1E-3/  \
--batch-size 64 --lr 0.005 --ard_init 1. --epoch-print 10

# All in cluster3
# - vbdl1 (ssr) => hyperparameter search
python mnist_compare_penalty.py --importance-method vbdl1 --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 0 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-8-vbd_l1-0/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10 && python mnist_compare_penalty.py --importance-method vbdl1 --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 1E-3 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-8-vbd_l1-1E-3/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10 && python mnist_compare_penalty.py --importance-method vbdl1 --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 1E-4 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-8-vbd_l1-1E-4/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10 && python mnist_compare_penalty.py --importance-method vbdl1 --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 1E-5 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-8-vbd_l1-1E-5/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10

# - vbdl1 (sdr) => hyperparameter search
python mnist_compare_penalty.py --importance-method vbd_l1_opposite --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 0 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-8-vbd_l1_opposite-0/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10 && python mnist_compare_penalty.py --importance-method vbd_l1_opposite --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 1E-4 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-8-vbd_l1_opposite-1E-4/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10 && python mnist_compare_penalty.py --importance-method vbd_l1_opposite --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 1E-5 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-8-vbd_l1_opposite-1E-5/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10 && python mnist_compare_penalty.py --importance-method vbd_l1_opposite --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 1E-6 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-8-vbd_l1_opposite-1E-6/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10

# p_b only flip 8
python mnist_compare_penalty.py --importance-method p_b --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 0 --epochs 600 \
--num-imgs 100 --save-dir ./result/1018-8-p_b/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10

# flip 8-3 with vae
python mnist_compare_penalty.py --importance-method p_b \
--gen_model VAEInpainter --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 0 --epochs 600 \
--num-imgs 100 --save-dir ./result/1028-vae-p_b/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10 && python mnist_compare_penalty.py --importance-method vbd_l1_opposite \
--gen_model VAEInpainter --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 0 --epochs 600 \
--num-imgs 100 --save-dir ./result/1028-vae-vbd_l1_opposite-0/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10 && python mnist_compare_penalty.py --importance-method vbdl1 \
--gen_model VAEInpainter --mode flip_two \
--the-digit 8 --to-digit 3 --reg-coef 0 --epochs 600 \
--num-imgs 100 --save-dir ./result/1028-vae-vbdl1-0/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10

# flip 8 with vae
python mnist_compare_penalty.py --importance-method p_b \
--gen_model VAEInpainter --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 0 --epochs 600 \
--num-imgs 100 --save-dir ./result/1028-8-vae-p_b/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10 && python mnist_compare_penalty.py --importance-method vbd_l1_opposite \
--gen_model VAEInpainter --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 0 --epochs 600 \
--num-imgs 100 --save-dir ./result/1028-8-vae-vbd_l1_opposite-0/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10 && python mnist_compare_penalty.py --importance-method vbdl1 \
--gen_model VAEInpainter --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 0 --epochs 600 \
--num-imgs 100 --save-dir ./result/1028-8-vae-vbdl1-0/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10

# Flip
python mnist_compare_penalty.py --importance-method vbdl1 \
--gen_model VAEInpainter --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 0.1 --epochs 600 \
--num-imgs 100 --save-dir ./result/1028-8-vae-vbdl1-0.1/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10 && python mnist_compare_penalty.py --importance-method vbdl1 \
--gen_model VAEInpainter --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 0.01 --epochs 600 \
--num-imgs 100 --save-dir ./result/1028-8-vae-vbdl1-0.01/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10 && python mnist_compare_penalty.py --importance-method vbdl1 \
--gen_model VAEInpainter --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 1E-3 --epochs 600 \
--num-imgs 100 --save-dir ./result/1028-8-vae-vbdl1-1E-3/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10

python mnist_compare_penalty.py --importance-method vbd_l1_opposite \
--gen_model VAEInpainter --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 0.1 --epochs 600 \
--num-imgs 100 --save-dir ./result/1028-8-vae-vbd_l1_opposite-0.1/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10 && python mnist_compare_penalty.py --importance-method vbd_l1_opposite \
--gen_model VAEInpainter --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 0.01 --epochs 600 \
--num-imgs 100 --save-dir ./result/1028-8-vae-vbd_l1_opposite-0.01/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10 && python mnist_compare_penalty.py --importance-method vbd_l1_opposite \
--gen_model VAEInpainter --mode flip_one \
--the-digit 8 --to-digit 3 --reg-coef 1E-3 --epochs 600 \
--num-imgs 100 --save-dir ./result/1028-8-vae-vbd_l1_opposite-1E-3/  \
--batch-size 64 --lr 0.01 --ard_init 0. --epoch-print 10