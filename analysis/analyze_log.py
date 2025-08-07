#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析扩散模型训练日志并生成图表
直接使用文本内容进行分析
"""

from draw import analyze_training_from_text

# 您提供的训练日志内容
log_content = """Loaded 12474 sequences for Diffusion model training (max_len=190).
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:62: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(
Starting Diffusion Model training...
Parameters: max_len=190, batch_size=64, epochs=200
Model parameters: 427,797
Early stopping: patience=30, min_improvement=0.001
Regularization: dropout=0.7, weight_decay=0.005, label_smoothing=0.2
Epoch 0 finished. Avg Loss: 1.8890, LR: 0.000030
  → New best loss: 1.8890
Epoch 1 finished. Avg Loss: 1.8227, LR: 0.000030
  → New best loss: 1.8227
Epoch 2 finished. Avg Loss: 1.8062, LR: 0.000030
  → New best loss: 1.8062
Epoch 3 finished. Avg Loss: 1.7851, LR: 0.000030
  → New best loss: 1.7851
Epoch 4 finished. Avg Loss: 1.7643, LR: 0.000030
  → New best loss: 1.7643
Epoch 5 finished. Avg Loss: 1.7687, LR: 0.000030
  → No improvement. Patience: 1/30
Epoch 6 finished. Avg Loss: 1.7676, LR: 0.000030
  → No improvement. Patience: 2/30
Epoch 7 finished. Avg Loss: 1.7581, LR: 0.000030
  → New best loss: 1.7581
[Eval] Reconstruction Accuracy:
  t=20: 0.9990
  t=50: 0.8135
  t=100: 0.3793
  t=200: 0.1378
[Sample] Example generated sequences:
  1: DLEDVEIFTT
  2: WMECHTTCMDHC
  3: GCRGGRIDYMEFNRVLIYTRMTPMDTTGKYMPKSDWQCWVMNAATMPPMDNCRWECLSLIIYIMPWKWVPNVHPKIKARNLPNVY
  4: EIKGSMKHPNTLQMQNHHQ
[Diversity] Valid sequences: 26/32
[Diversity] Generation diversity: 1.0000, Avg length: 27.0
Epoch 8 finished. Avg Loss: 1.7423, LR: 0.000030
  → New best loss: 1.7423
Epoch 9 finished. Avg Loss: 1.7710, LR: 0.000030
  → No improvement. Patience: 1/30
Epoch 10 finished. Avg Loss: 1.7535, LR: 0.000030
  → No improvement. Patience: 2/30
Epoch 11 finished. Avg Loss: 1.7734, LR: 0.000030
  → No improvement. Patience: 3/30
Epoch 12 finished. Avg Loss: 1.7492, LR: 0.000030
  → No improvement. Patience: 4/30
Epoch 13 finished. Avg Loss: 1.7753, LR: 0.000030
  → No improvement. Patience: 5/30
Epoch 14 finished. Avg Loss: 1.7598, LR: 0.000030
  → No improvement. Patience: 6/30
Epoch 15 finished. Avg Loss: 1.7438, LR: 0.000030
  → No improvement. Patience: 7/30
[Eval] Reconstruction Accuracy:
  t=20: 1.0000
  t=50: 0.8191
  t=100: 0.3888
  t=200: 0.1434
[Sample] Example generated sequences:
  1: KAIMESPMFSCYKYC
  2: LCGSVRHVWQARFIKHIYGK
  3: SMSWALDVPQHTDDADWVWNTV
  4: SMMAY
[Diversity] Valid sequences: 31/32
[Diversity] Generation diversity: 1.0000, Avg length: 21.3
Epoch 16 finished. Avg Loss: 1.7402, LR: 0.000030
  → New best loss: 1.7402
Epoch 17 finished. Avg Loss: 1.7445, LR: 0.000030
  → No improvement. Patience: 1/30
Epoch 18 finished. Avg Loss: 1.7464, LR: 0.000030
  → No improvement. Patience: 2/30
Epoch 19 finished. Avg Loss: 1.7499, LR: 0.000030
  → No improvement. Patience: 3/30
Epoch 20 finished. Avg Loss: 1.7633, LR: 0.000030
  → No improvement. Patience: 4/30
Epoch 21 finished. Avg Loss: 1.7613, LR: 0.000030
  → No improvement. Patience: 5/30
Epoch 22 finished. Avg Loss: 1.7492, LR: 0.000030
  → No improvement. Patience: 6/30
Epoch 23 finished. Avg Loss: 1.7401, LR: 0.000030
  → No improvement. Patience: 7/30
[Eval] Reconstruction Accuracy:
  t=20: 0.9980
  t=50: 0.8230
  t=100: 0.3878
  t=200: 0.1424
[Sample] Example generated sequences:
  1: RGCWMFTVAVLGCFC
  2: EMMVGWNMERWRDRPDKEPGNYRDQVCQIFSV
  3: MHCKEINII
  4: PCNPPCATQGKMLW
[Diversity] Valid sequences: 28/32
[Diversity] Generation diversity: 1.0000, Avg length: 27.6
Epoch 24 finished. Avg Loss: 1.7373, LR: 0.000030
  → New best loss: 1.7373
Epoch 25 finished. Avg Loss: 1.7344, LR: 0.000030
  → New best loss: 1.7344
Epoch 26 finished. Avg Loss: 1.7464, LR: 0.000030
  → No improvement. Patience: 1/30
Epoch 27 finished. Avg Loss: 1.7325, LR: 0.000030
  → New best loss: 1.7325
Epoch 28 finished. Avg Loss: 1.7369, LR: 0.000030
  → No improvement. Patience: 1/30
Epoch 29 finished. Avg Loss: 1.7212, LR: 0.000030
  → New best loss: 1.7212
Epoch 30 finished. Avg Loss: 1.7033, LR: 0.000030
  → New best loss: 1.7033
Epoch 31 finished. Avg Loss: 1.6966, LR: 0.000030
  → New best loss: 1.6966
[Eval] Reconstruction Accuracy:
  t=20: 0.9967
  t=50: 0.7447
  t=100: 0.3122
  t=200: 0.1063
[Sample] Example generated sequences:
  1: LMLCTKEAQMGYMEMFQNWVH
  2: WKMVKWPWEYDWK
  3: MNKNAEDFDTLN
  4: QDFRQGQ
[Diversity] Valid sequences: 22/32
[Diversity] Generation diversity: 1.0000, Avg length: 14.2
Epoch 32 finished. Avg Loss: 1.7223, LR: 0.000030
  → No improvement. Patience: 1/30
Epoch 33 finished. Avg Loss: 1.7032, LR: 0.000030
  → No improvement. Patience: 2/30
Epoch 34 finished. Avg Loss: 1.6861, LR: 0.000030
  → New best loss: 1.6861
Epoch 35 finished. Avg Loss: 1.6763, LR: 0.000030
  → New best loss: 1.6763
Epoch 36 finished. Avg Loss: 1.6579, LR: 0.000030
  → New best loss: 1.6579
Epoch 37 finished. Avg Loss: 1.6559, LR: 0.000030
  → New best loss: 1.6559
Epoch 38 finished. Avg Loss: 1.6699, LR: 0.000030
  → No improvement. Patience: 1/30
Epoch 39 finished. Avg Loss: 1.6417, LR: 0.000030
  → New best loss: 1.6417
[Eval] Reconstruction Accuracy:
  t=20: 0.9934
  t=50: 0.7625
  t=100: 0.3349
  t=200: 0.1316
[Sample] Example generated sequences:
  1: MCLCCCHTESFIADVNVSMHTSLFQGYIPSHHWFSHVMMDIDGIFISCPPYVDEC
  2: PQFHN
  3: RNSKPWWK
  4: CRDCEQEHAVPTL
[Diversity] Valid sequences: 26/32
[Diversity] Generation diversity: 1.0000, Avg length: 17.5
Epoch 40 finished. Avg Loss: 1.6271, LR: 0.000030
  → New best loss: 1.6271
Epoch 41 finished. Avg Loss: 1.6191, LR: 0.000030
  → New best loss: 1.6191
Epoch 42 finished. Avg Loss: 1.6341, LR: 0.000030
  → No improvement. Patience: 1/30
Epoch 43 finished. Avg Loss: 1.6153, LR: 0.000030
  → New best loss: 1.6153
Epoch 44 finished. Avg Loss: 1.6280, LR: 0.000030
  → No improvement. Patience: 1/30
Epoch 45 finished. Avg Loss: 1.6114, LR: 0.000030
  → New best loss: 1.6114
Epoch 46 finished. Avg Loss: 1.6116, LR: 0.000030
  → No improvement. Patience: 1/30
Epoch 47 finished. Avg Loss: 1.6094, LR: 0.000030
  → New best loss: 1.6094
[Eval] Reconstruction Accuracy:
  t=20: 0.9934
  t=50: 0.7563
  t=100: 0.3431
  t=200: 0.1257
[Sample] Example generated sequences:
  1: LAGMNV
  2: IEGDDDM
  3: PINTTTHPRSAFGNDRYPVSSGFW
  4: RGEGRNDILVIQVQSDADVPPIDASPGHDPIPAISLMVALSINSV
[Diversity] Valid sequences: 28/32
[Diversity] Generation diversity: 1.0000, Avg length: 18.8
Epoch 48 finished. Avg Loss: 1.6222, LR: 0.000030
  → No improvement. Patience: 1/30
Epoch 49 finished. Avg Loss: 1.6191, LR: 0.000030
  → No improvement. Patience: 2/30
Epoch 50 finished. Avg Loss: 1.6168, LR: 0.000030
  → No improvement. Patience: 3/30
Epoch 51 finished. Avg Loss: 1.6113, LR: 0.000030
  → No improvement. Patience: 4/30
Epoch 52 finished. Avg Loss: 1.6063, LR: 0.000030
  → New best loss: 1.6063
Epoch 53 finished. Avg Loss: 1.6052, LR: 0.000030
  → New best loss: 1.6052
Epoch 54 finished. Avg Loss: 1.6080, LR: 0.000030
  → No improvement. Patience: 1/30
Epoch 55 finished. Avg Loss: 1.6122, LR: 0.000030
  → No improvement. Patience: 2/30
[Eval] Reconstruction Accuracy:
  t=20: 0.9961
  t=50: 0.7678
  t=100: 0.3477
  t=200: 0.1418
[Sample] Example generated sequences:
  1: TQMKTDIV
  2: WDYVGHPHTI
  3: KHHNICRHTRWYLPITLLI
  4: SFKGKLGERDIR
[Diversity] Valid sequences: 21/32
[Diversity] Generation diversity: 1.0000, Avg length: 17.5
Epoch 56 finished. Avg Loss: 1.6090, LR: 0.000030
  → No improvement. Patience: 3/30
Epoch 57 finished. Avg Loss: 1.5946, LR: 0.000030
  → New best loss: 1.5946
Epoch 58 finished. Avg Loss: 1.6042, LR: 0.000030
  → No improvement. Patience: 1/30
Epoch 59 finished. Avg Loss: 1.6174, LR: 0.000030
  → No improvement. Patience: 2/30
Epoch 60 finished. Avg Loss: 1.6150, LR: 0.000030
  → No improvement. Patience: 3/30
Epoch 61 finished. Avg Loss: 1.6249, LR: 0.000030
  → No improvement. Patience: 4/30
Epoch 62 finished. Avg Loss: 1.6184, LR: 0.000030
  → No improvement. Patience: 5/30
Epoch 63 finished. Avg Loss: 1.6205, LR: 0.000030
  → No improvement. Patience: 6/30
[Eval] Reconstruction Accuracy:
  t=20: 0.9941
  t=50: 0.7618
  t=100: 0.3454
  t=200: 0.1188
[Sample] Example generated sequences:
  1: FAQRDKGGFMFPLCPGDCEQF
  2: TFSHEVKFYDVPGATTWMFFVSLHPVPYKG
  3: HVAEKVKTMWVSGNMMGSKMTQFC
  4: WMFDEQDRWVRPSCWGGGNPFAN
[Diversity] Valid sequences: 23/32
[Diversity] Generation diversity: 1.0000, Avg length: 19.7
Epoch 64 finished. Avg Loss: 1.6215, LR: 0.000030
  → No improvement. Patience: 7/30
Epoch 65 finished. Avg Loss: 1.6119, LR: 0.000030
  → No improvement. Patience: 8/30
Epoch 66 finished. Avg Loss: 1.5938, LR: 0.000030
  → No improvement. Patience: 9/30
Epoch 67 finished. Avg Loss: 1.6112, LR: 0.000030
  → No improvement. Patience: 10/30
Epoch 68 finished. Avg Loss: 1.6026, LR: 0.000030
  → No improvement. Patience: 11/30
Epoch 69 finished. Avg Loss: 1.5972, LR: 0.000030
  → No improvement. Patience: 12/30
Epoch 70 finished. Avg Loss: 1.6143, LR: 0.000030
  → No improvement. Patience: 13/30
Epoch 71 finished. Avg Loss: 1.6064, LR: 0.000030
  → No improvement. Patience: 14/30
[Eval] Reconstruction Accuracy:
  t=20: 0.9974
  t=50: 0.7553
  t=100: 0.3345
  t=200: 0.1132
[Sample] Example generated sequences:
  1: DQCNEMNYTEKDNGDMKPQEAEYDADFCKEGMYY
  2: RKDCFYPRPVFKVSMKVTGESL
  3: YGFDFGHLCYLEKCVDMSFILFVWGYG
  4: YSCHWWKTYDDCDHHYQNW
[Diversity] Valid sequences: 24/32
[Diversity] Generation diversity: 1.0000, Avg length: 21.4
Epoch 72 finished. Avg Loss: 1.6114, LR: 0.000030
  → No improvement. Patience: 15/30
Epoch 73 finished. Avg Loss: 1.5965, LR: 0.000030
  → No improvement. Patience: 16/30
Epoch 74 finished. Avg Loss: 1.5853, LR: 0.000030
  → New best loss: 1.5853
Epoch 75 finished. Avg Loss: 1.5989, LR: 0.000030
  → No improvement. Patience: 1/30
Epoch 76 finished. Avg Loss: 1.5979, LR: 0.000030
  → No improvement. Patience: 2/30
Epoch 77 finished. Avg Loss: 1.6049, LR: 0.000030
  → No improvement. Patience: 3/30
Epoch 78 finished. Avg Loss: 1.6080, LR: 0.000030
  → No improvement. Patience: 4/30
Epoch 79 finished. Avg Loss: 1.5980, LR: 0.000030
  → No improvement. Patience: 5/30
[Eval] Reconstruction Accuracy:
  t=20: 0.9977
  t=50: 0.7684
  t=100: 0.3559
  t=200: 0.1204
[Sample] Example generated sequences:
  1: AWSTANVK
  2: DMAKPMNQVHISC
  3: MESTHEMT
  4: CEDKLAASCGHWYMFL
[Diversity] Valid sequences: 23/32
[Diversity] Generation diversity: 1.0000, Avg length: 23.7
Epoch 80 finished. Avg Loss: 1.6026, LR: 0.000030
  → No improvement. Patience: 6/30
Epoch 81 finished. Avg Loss: 1.6054, LR: 0.000030
  → No improvement. Patience: 7/30
Epoch 82 finished. Avg Loss: 1.6072, LR: 0.000030
  → No improvement. Patience: 8/30
Epoch 83 finished. Avg Loss: 1.6016, LR: 0.000015
  → No improvement. Patience: 9/30
Epoch 84 finished. Avg Loss: 1.5953, LR: 0.000015
  → No improvement. Patience: 10/30
Epoch 85 finished. Avg Loss: 1.6026, LR: 0.000015
  → No improvement. Patience: 11/30
Epoch 86 finished. Avg Loss: 1.6110, LR: 0.000015
  → No improvement. Patience: 12/30
Epoch 87 finished. Avg Loss: 1.5937, LR: 0.000015
  → No improvement. Patience: 13/30
[Eval] Reconstruction Accuracy:
  t=20: 0.9957
  t=50: 0.7770
  t=100: 0.3684
  t=200: 0.1276
[Sample] Example generated sequences:
  1: QKKVQTWCGFICFHMVCRVINLRVIAVSTLQREETVAMRWYTF
  2: DKVPTMNPMRGIPHKK
  3: AIMLVHVGPDLKPMPKEGLHNTWPPKVDSKNYCDKVMISMRCPWVVTRNGI
  4: LNMCSPSHVILEMN
[Diversity] Valid sequences: 27/32
[Diversity] Generation diversity: 1.0000, Avg length: 24.1
Epoch 88 finished. Avg Loss: 1.6073, LR: 0.000015
  → No improvement. Patience: 14/30
Epoch 89 finished. Avg Loss: 1.5993, LR: 0.000015
  → No improvement. Patience: 15/30
Epoch 90 finished. Avg Loss: 1.6093, LR: 0.000015
  → No improvement. Patience: 16/30
Epoch 91 finished. Avg Loss: 1.5979, LR: 0.000015
  → No improvement. Patience: 17/30
Epoch 92 finished. Avg Loss: 1.6166, LR: 0.000008
  → No improvement. Patience: 18/30
Epoch 93 finished. Avg Loss: 1.6016, LR: 0.000008
  → No improvement. Patience: 19/30
Epoch 94 finished. Avg Loss: 1.6022, LR: 0.000008
  → No improvement. Patience: 20/30
Epoch 95 finished. Avg Loss: 1.6059, LR: 0.000008
  → No improvement. Patience: 21/30
[Eval] Reconstruction Accuracy:
  t=20: 0.9954
  t=50: 0.7641
  t=100: 0.3490
  t=200: 0.1178
[Sample] Example generated sequences:
  1: VCNPMCVAGGH
  2: DDDAGTARIQYDKGEVNTLMNNNMMMTKHEIGGWGEWNTGQWTYDTETAILKCKEPS
  3: MEQTGSEWILF
  4: KCEINLFSFHEKGLKDSM
[Diversity] Valid sequences: 23/32
[Diversity] Generation diversity: 1.0000, Avg length: 21.8
Epoch 96 finished. Avg Loss: 1.6087, LR: 0.000008
  → No improvement. Patience: 22/30
Epoch 97 finished. Avg Loss: 1.6074, LR: 0.000008
  → No improvement. Patience: 23/30
Epoch 98 finished. Avg Loss: 1.6074, LR: 0.000008
  → No improvement. Patience: 24/30
Epoch 99 finished. Avg Loss: 1.6034, LR: 0.000008
  → No improvement. Patience: 25/30
Epoch 100 finished. Avg Loss: 1.5989, LR: 0.000008
  → No improvement. Patience: 26/30
Epoch 101 finished. Avg Loss: 1.6133, LR: 0.000004
  → No improvement. Patience: 27/30
Epoch 102 finished. Avg Loss: 1.5959, LR: 0.000004
  → No improvement. Patience: 28/30
Epoch 103 finished. Avg Loss: 1.5965, LR: 0.000004
  → No improvement. Patience: 29/30
[Eval] Reconstruction Accuracy:
  t=20: 0.9974
  t=50: 0.7734
  t=100: 0.3457
  t=200: 0.1329
[Sample] Example generated sequences:
  1: KMSDHPHNHTNKTKRTRGTINKAQTAVPSKYFEQKDALRYAFTVYQGDGGPNPFPKHRCDKKE
  2: CLQDPVS
  3: WLYFPMFTSGTNDMCRSITPTWINTEDNMLKDC
  4: DQYSFCMYMSS
[Diversity] Valid sequences: 23/32
[Diversity] Generation diversity: 1.0000, Avg length: 26.5
Epoch 104 finished. Avg Loss: 1.6063, LR: 0.000004
  → No improvement. Patience: 30/30

Early stopping triggered after 105 epochs
Best loss: 1.5853
Loaded best model with loss: 1.5853

Training finished. Model saved to models/diffusion_model_transformer.pth
Total training time: 591.49 seconds."""

if __name__ == "__main__":
    print("开始分析扩散模型训练日志...")
    print("=" * 60)

    # 分析训练日志并生成图表
    log_data = analyze_training_from_text(log_content, "./results/")

    print("\n分析完成！生成的文件包括：")
    print("1. training_metrics.png - 训练过程综合指标图")
    print("2. reconstruction_heatmap.png - 重构准确率热力图")
    print("3. early_stopping_analysis.png - 早停机制分析图")
    print("4. training_summary.txt - 训练总结报告")

    print("\n这些图表可以直接用于论文中！")
