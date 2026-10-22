# DeepSilencer: A Novel Deep Learning Model for Predicting siRNA Knockdown Efficiency

#### DeepSilencer: A Novel Deep Learning Model for Predicting siRNA Knockdown Efficiency

Wangdan Liao<sup>1</sup> and Weidong Wang<sup>1,2*</sup>

> 1School of Biological Science and Medical Engineering, Beihang University, 37 Xueyuan Road, Beijing, 100191, China.

> 2*Medical Innovation Research Division, Chinese PLA General Hospital, 28 Fuxing Road, Beijing, 100853, China.

*Corresponding author(s). E-mail(s): wangwd301@126.com; Contributing authors: liaowangdan@buaa.edu.cn;

## Abstract

**Background:** Small interfering RNA (siRNA) is a promising therapeutic agent due to its ability to silence disease-related genes via RNA interference. While traditional machine learning and early deep learning methods have made progress in predicting siRNA efficacy, there remains significant room for improvement. Advanced deep learning techniques can enhance prediction accuracy, reducing the reliance on extensive wet-lab experiments and accelerating the identification of effective siRNA sequences. This approach also provides deeper insights into the mechanisms of siRNA efficacy, facilitating more targeted and efficient therapeutic strategies.

**Methods:** We introduce DeepSilencer, an innovative deep learning model designed to predict siRNA knockdown efficiency. DeepSilencer utilizes advanced neural network architectures to capture the complex features of siRNA sequences. Our key contributions include a specially designed deep learning model, an innovative online data sampling method, and an improved loss function tailored for siRNA prediction. These enhancements collectively boost the model’s prediction accuracy and robustness.

**Results:** Extensive evaluations on multiple test sets demonstrate that DeepSilencer achieves state-of-the-art performance using only siRNA sequences and basic physicochemical properties. Our model surpasses several other methods and shows superior predictive performance, particularly when incorporating thermodynamic parameters.

**Conclusion:** The advancements in data sampling, model design, and loss function significantly enhance the predictive capabilities of DeepSilencer. These improvements underscore its potential to advance RNAi therapeutic design and development, offering a powerful tool for researchers and clinicians.

**Keywords:** RNAi, siRNA, gene knockdown, deep learning

## 1 Background

RNA interference (RNAi) technology has emerged as a promising therapeutic approach by enabling the sequence-specific silencing of disease-related genes. Small interfering RNA (siRNA) is central to this process, being incorporated into the RNA-induced silencing complex (RISC), which includes proteins such as Argonaute-2 (Ago-2), Dicer, and TRBP[1], [2] and [3]. The antisense strand of siRNA guides RISC to the target mRNA, where Ago-2 mediates cleavage, resulting in gene knockdown[4]. This mechanism allows siRNA to target traditionally undruggable genes, significantly expanding potential therapeutic targets.

Designing highly efficient siRNAs remains challenging. Effective siRNAs must avoid triggering the innate immune system, achieve high specificity in target cleavage, and minimize off-target effects. Consequently, designing effective siRNA is crucial for RNAi therapeutics, prompting the development of various computational tools. Tools like OligoWalk, siRNAPred, and i-score have limitations due to dataset biases, prediction methods, and off-target effects[5], [6] and [7].

Early machine learning methods, such as LASSO regression, support vector machines (SVM), and linear regression, improved prediction accuracy but were limited by their simple structures, which failed to capture complex features[8], [9]. With the advent of deep learning, researchers began leveraging advanced models for better understanding siRNA sequences.

Deep learning brought significant improvements. CNNs and DNNs automatically extracted features from siRNA sequences, enhancing accuracy[10]. Graph Neural Networks (GNNs) modeled siRNA-mRNA interactions, incorporating thermodynamic parameters[11]. Transformer-based models have excelled in natural language processing and genomics due to their ability to handle sequential data through attention mechanisms[12]. Models like BERT-siRNA leverage pre-training on extensive human genomic sequences to enhance siRNA efficacy prediction[13], [14]. OligoFormer combines Transformers with LSTM and convolutional layers to capture deep sequence features and complex siRNA-mRNA interaction patterns[15], [16] and [17].

Despite these advancements, existing models face limitations in interpretability, usability, robustness, and data utilization. In response, we propose DeepSilencer, a deep learning model specifically designed for predicting siRNA knockdown efficiency. Our contributions include:

1. **Innovative Model Design** : Combines transformer and CNN modules to effectively capture deep features of siRNA sequences, enhancing prediction accuracy and robustness without pre-training.

2. **Versatile Data Input** : Supports siRNA sequences alone or combined with thermodynamic features.

3. **Enhanced Loss Function** : Improved loss functions tailored for siRNA prediction tasks, boosting accuracy and robustness.

4. **Comprehensive Improvements** : Achieves state-of-the-art results using only siRNA sequences and simple physicochemical properties.

In summary, our work builds on existing methods’ strengths while addressing their limitations, offering a robust solution for siRNA knockdown efficiency prediction.

## 2 Methods

We have implemented a series of innovative approaches to improve the accuracy and robustness of siRNA silencing efficiency prediction. Initially, in response to the limited quantity of siRNA data, we introduced a novel dynamic random data pairing mechanism, namely Selective Pair Sampling, to fully exploit the potential of the available data. Subsequently, given the challenge of aligning different publicly available datasets on siRNA silencing efficiency, we designed a novel multi-task system named TriTask Learning Framework. To further refine this framework, we crafted an improved loss function that significantly boosts the model’s accuracy and robustness. Finally, we designed DeepSilencer, a model specifically tailored for RNA sequence analysis. By integrating Transformer and Convolutional Neural Network (CNN) architectures, DeepSilencer effectively captures the intricate features of siRNA sequences, thereby improving prediction accuracy and robustness without the need for pre-training. Detailed introductions of these components will be presented in the following sections.

### 2.1 Data Processing

We utilized the datasets summarized by OligoFormer, which aggregated nine studies, resulting in a total of 3,714 siRNAs and 75 mRNAs[18], [19], [20], [21],[22], [23], [24] and [25]. These datasets were organized into three categories: the Huesken dataset, the Takayuki dataset, and a mixed set (Mixset) comprising the remaining data. The inhibition efficacy of siRNAs was normalized to a range of 0 to 100%, with a 70% inhibition threshold distinguishing positive from negative siRNAs.

To ensure consistency across datasets, all siRNA sequences were truncated to 19 nucleotides. This standardization facilitated uniform processing and feature extraction. OligoFormer extracted thermodynamic parameters for each siRNA, including stability and binding affinity, essential for predicting siRNA efficacy. Detailed thermodynamic parameters are provided in Table 1. Subsequently, considering the significance of these parameters, we conducted additional analyses on them. Recognizing the complexity of feature interactions, we employed a random forest model to fit the silencing efficiency solely based on thermodynamic features and obtained feature importance scores. Although the feature importance analysis based on the random forest model indicated that the importance levels of some features were relatively low, we discovered that removing these features would affect the performance of our model. Therefore, after careful consideration, we ultimately decided to retain these features to maintain the overall performance and stability of the model.

Our model, DeepSilencer, employs these processed datasets for training and validation without requiring complex pre-training. It supports two prediction modes: one using only siRNA sequence data and the other using both siRNA sequences and thermodynamic parameters. In subsequent experiments, we will evaluate the model’s performance under both conditions.

**Table 1** : Thermodynamic Parameters for siRNA-mRNA Binding

[Table 1](../assets/table/table-1.csv)

| Parameter | Description | Feature Contribution |
| --- | --- | --- |
| ΔΔG°_ends | End free energy difference | 0.264 |
| ΔG°_1 | Free energy change of the first base pair | 0.039 |
| ΔH°_1 | Enthalpy change of the first base pair | 0.018 |
| U_1 | Uracil content of the first base pair | 0.023 |
| G_1 | Guanine content of the first base pair | 0.003 |
| ΔH°_all | Total enthalpy change of all base pairs | 0.218 |
| U_all | Total uracil content of all base pairs | 0.052 |
| UU_1 | Uracil content of the first base pair and its neighbors | 0.003 |
| G_all | Total guanine content of all base pairs | 0.054 |
| GG_1 | Guanine content of the first base pair and its neighbors | 0.002 |
| GC_1 | Guanine and cytosine content of the first base pair | 0.001 |
| GG_all | Total guanine content of all base pairs | 0.033 |
| ΔG°_2 | Free energy change of the second base pair | 0.046 |
| UA_all | Total uracil and adenine content of all base pairs | 0.022 |
| U_2 | Uracil content of the second base pair | 0.007 |
| C_1 | Cytosine content of the first base pair | 0.004 |
| CC_all | Total cytosine content of all base pairs | 0.033 |
| ΔG°_18 | Free energy change of the eighteenth base pair | 0.037 |
| CC_1 | Cytosine content of the first base pair and its neighbors | 0.004 |
| GC_all | Total guanine and cytosine content of all base pairs | 0.031 |
| CG_1 | Cytosine and guanine content of the first base pair | 0.007 |
| ΔG°_13 | Free energy change of the thirteenth base pair | 0.051 |
| UU_all | Total uracil content of all base pairs | 0.028 |
| A_19 | Adenine content of the nineteenth base pair | 0.013 |

### 2.2 Selective Pair Sampling

Due to the limited availability of siRNA data, with the largest dataset containing fewer than 3000 siRNAs, it is crucial to maximize the potential of the available data and enhance the model’s ranking capability. We designed a novel pair loss-based approach for siRNA silencing efficiency prediction[26]. By transforming each training sample from a single siRNA to a randomly paired siRNA pair, this method not only increases data diversity but also improves the model’s ranking ability through pair loss.

To achieve this, we implemented an innovative dynamic random data pairing mechanism to sample siRNA pairs that meet specific criteria (see Algorithm 1). This method, called **Selective Pair Sampling** , ensures that the sampled siRNA pairs have label differences within a predefined range, thereby enhancing the model’s ability to learn from subtle differences in siRNA efficacy.

The **Selective Pair Sampling** method operates as follows:

1. **Random Initial Selection** : For each data retrieval operation, an initial siRNA sequence ( _s_ 1) with a length of 19 nucleotides and its corresponding label ( _y_ 1) is selected from the dataset.

2. **Conditional Pairing** : The method then enters a loop where it randomly selects a second siRNA sequence ( _s_ 2) and its label ( _y_ 2). This loop continues until a pair is found where the absolute difference between their labels ( _y_ 1 and _y_ 2) falls within the range of _α_ 1 to _α_ 2. This condition ensures that the selected pairs exhibit a moderate difference in inhibition efficiency, which is crucial for training models to discern subtle variations.

3. **Data Preparation** : Once a suitable pair is identified, both sequences are tokenized using a tokenizer. The tokenized sequences are padded and truncated to a maximum length of 21 nucleotides, with a `[CLS]` token added at the beginning and a `[SEQ]` token at the end.

4. **Return Values** : The method returns the tokenized sequences, their corresponding normalized silencing efficiencies, thermodynamic parameters(TR), and their class labels.

**Algorithm 1** Selective Pair Sampling

~~~~text
1: Input: Dataset D with siRNA sequences (s), thermodynamic parameters (tr), and labels (y)
2: Output: Paired siRNA sequences and features
3: for each data retrieval operation do
4:     s_1, tr_1, y_1 ← Randomly select an siRNA sequence and its label from D
5:     while True do
6:         s_2, tr_2, y_2 ← Randomly select another siRNA sequence and its label from D
7:         if α_1 < |y_1 − y_2| < α_2 then
8:             break
9:         end if
10:    end while
11:    Sample the pair (s_1, s_2), their thermodynamic parameters (tr_1, tr_2), and their respective labels (y_1, y_2)
12: end for
~~~~

### 2.3 Multi-Task Learning for siRNA Silencing Efficiency Prediction

Existing publicly available data on siRNA silencing efficiency are derived from various experimental measurements. However, due to differences in experimental methodologies and measurement techniques, aligning the values across different datasets can be challenging. In the OligoFormer study, the authors addressed this issue by converting continuous values into categorical labels. Specifically, the inhibition efficacy or activity of siRNAs in all datasets was normalized to a range from 0 to 100%, and a threshold of 70% maximum inhibition was used to classify siRNAs as positive or negative.

However, this approach introduces a problem: siRNAs with similar efficacy near the classification boundary may end up with distinctly different labels. To address this, we designed a novel multi-task system, named **Tri-Task Learning Framework** , which consists of three tasks[27]:

1. **Classification Task** : Similar to OligoFormer, this task classifies siRNAs based on a predefined threshold.

2. **Regression Task** : This task directly predicts the normalized silencing efficiency.
3. **Contrastive Task** : This task predicts the difference in silencing efficiency between pairs of siRNAs, aiming to improve the model’s ranking capability.

The total loss for the Tri-Task Learning Framework is a combination of the losses from the three tasks:

- **Classification Loss** ( _L_ cls): This is typically a binary cross-entropy loss for classifying siRNAs as positive or negative. This loss is specifically for siRNA1 and does not include siRNA2.

- **Regression Loss** ( _L_ reg): This is typically a Smooth L1 Loss (also known as Huber Loss) for predicting the normalized silencing efficiency. This loss includes both siRNA1 and siRNA2.

- **Contrastive Loss** ( _L_ cont): This loss measures the ability of the model to predict the difference in silencing efficiency between pairs of siRNAs. One common approach is to use a margin-based contrastive loss. This loss has a coefficient of 2.

The Smooth L1 Loss, also known as Huber Loss, is defined as:

[Formula on PDF page 6 (10)](../assets/figure/formula-p0006-009.jpg)

Here, _β_ is a hyperparameter that determines the point at which the loss function transitions from quadratic to linear. In our experiments, we set _β_ = 2.4[28].

This loss function is less sensitive to outliers compared to the Mean Squared Error (MSE) loss, making it more robust for regression tasks where outliers might be present. It combines the best properties of both L1 and L2 losses by being quadratic when the error is small and linear when the error is large.

Given two siRNA sequences, siRNA1 and siRNA2, the model, DeepSilencer, predicts their respective silencing efficiencies, ˆy1 and ˆy2. The output of the contrastive model is defined as the difference between these predictions:

[Predicted difference](../assets/figure/predicted-difference.jpg)

where ∆ˆ _y_ represents the predicted difference in silencing efficiency between the two siRNAs.

The true difference in silencing efficiencies for the siRNA pair is denoted as ∆ _y_ = _y_ 1 _− y_ 2, where _y_ 1 and _y_ 2 are the actual silencing efficiencies of siRNA1 and siRNA2, respectively.

The contrastive loss is then computed using the Smooth L1 loss function, which is

defined as:

[Contrastive loss](../assets/figure/contrastive-loss.jpg)

This formulation allows the model to effectively learn to predict the relative differences in silencing efficiency between pairs of siRNAs, which is crucial for its overall performance.

The total loss _L_ total can be expressed as:

[Total loss](../assets/figure/total-loss.jpg)

Here, the coefficient 2 in front of _L_ cont is manually tuned based on our validation experiments. The initial motivation was to enhance the model’s ranking capability.

### 2.4 Model Architecture

Our model, **DeepSilencer** , integrates both Transformer and Convolutional Neural Network (CNN) architectures, specifically designed for RNA sequence analysis. The structure of the model is shown in Fig 1. The Transformer, through its self-attention mechanism, excels at capturing global dependencies within sequences, demonstrating strong modeling capabilities in both natural language processing and bioinformatics. However, siRNA sequences are typically short, and the available data is limited. Additionally, RNA molecular interactions frequently exhibit highly localized characteristics, making it challenging for a Transformer alone to sufficiently capture these local dependencies.

For instance, the self-attention mechanism of the Transformer performs exceptionally well when handling long sequences because it can simultaneously attend to all positions within the sequence at each layer. However, for short sequences like siRNA, an over-reliance on global information may lead to the model overlooking critical local features. Moreover, Transformer models generally require large datasets to effectively learn the complex relationships between sequences, and the limited size of siRNA datasets might hinder the model’s generalization capabilities.

In contrast, Convolutional Neural Networks (CNNs), with their local receptive fields and weight-sharing mechanisms, are adept at extracting local features and possess a strong inductive bias. Studies have shown that CNNs exhibit robust performance when dealing with small-scale datasets. For example, CoAtNet combines convolutional and Transformer modules to achieve significant performance improvements on large-scale image datasets like ImageNet[30]. Similarly, Astroformer demonstrates that combining convolutional and Transformer structures and training from scratch can achieve excellent results on small-scale datasets[31].

Building on this foundation, we designed the **DeepSilencer** network, which leverages the strengths of both convolutional and Transformer architectures. The overall structure is as follows:

[Figure 1](../assets/figure/figure-1.jpg)

**Fig. 1** : **DeepSilencer structure.** DeepSilencer is primarily composed of a Transformer encoder, consisting of multiple layers of transformers, and a ConvNeXt1D Encoder, formed by stacking multiple ConvNeXtBlocks in series[29]. If thermodynamic parameters are included, a projection layer made of multilayer perceptrons(MLP) is used to transform the thermodynamic parameters into the hidden layer’s dimensions. This transformation acts as a bias, added to the hidden vectors of each token obtained from the Transformer. Afterwards, the ConvNeXt Encoder performs dimensionality reduction on both the sequence and hidden layer dimensions of the sequence’s hidden vectors. These vectors are then flattened and passed through two task-specific layers, each composed of MLP, to predict both the normalized knockdown efficiency value and the knockdown efficiency classification. Notably, the predicted efficiency value is related to both regression and contrastive tasks. .

The first part of the model consists of a Transformer, which integrates global sequence information from siRNA. We utilize the T5 Encoder as the foundational structure, adjusting its configuration parameters as needed[30]. The T5 Encoder is a Transformer model that uses relative position encoding, here, we only utilized its architecture. Specifically, we configured a hidden layer dimension of 128, with 4 Transformer encoder layers, 4 attention heads, and a feed-forward network dimension of 128 _×_ 4.

The latter part of the model comprises several one-dimensional ConvNeXtBlocks, which further enhance the integration of local information. ConvNeXt is an enhanced convolutional neural network architecture that improves performance through various innovations. ConvNeXtBlocks primarily consist of downsampling layers, depth-wise convolutions, LayerNorm, multilayer perceptrons (MLPs, equivalent to Feed Forward Layers in this context), and residual connections. In our implementation, we made a slight modification by substituting the original 2D convolutions with 1D convolutions. We incorporated multiple ConvNeXtBlocks to progressively extract and integrate local features:

- The convolutional operations are divided into three stages, with downsampling achieved by setting the convolutional stride to 2 at the end of the first two stages.

- The first convolutional stage reduces the hidden layer dimension from 128 to 64, followed by a stride-2 convolutional layer for further downsampling. This stage contains 2 ConvNeXtBlocks.

- The second convolutional stage reduces the hidden layer dimension from 64 to 32, employing multiple stride-1 and stride-2 convolutional layers to gradually extract features. This stage comprises 3 ConvNeXtBlocks.

- The final convolutional stage maintains the dimension at 32 and includes 1 ConvNeXtBlock.

After the convolutional stages, a flattening operation merges the hidden and sequence dimensions before passing the features to the task-specific layers. Finally, the model includes two task-specific fully connected layers for predicting siRNA knockdown efficiency and classification:

- **Knockdown Efficiency Prediction Layer** : This layer consists of a dropout layer, a linear layer, and a GELU activation function, ultimately outputting a scalar value.

- **Knockdown Efficiency Classification Layer** : This layer, similar in structure, outputs the probabilities for two classes.

[Figure 2](../assets/figure/figure-2.jpg)

**Fig. 2** : (a) Transformer Encoder: This part of the model integrates global sequence information from siRNA using the T5 Encoder architecture. The configuration includes a hidden layer dimension of 128, 4 Transformer encoder layers, 4 attention heads, and a feed-forward network dimension of 128 × 4. (b) ConvNeXt Encoder: This part of the model enhances the integration of local sequence information using several onedimensional ConvNeXtBlocks. The ConvNeXt Encoder are divided into three stages: the first stage reduces the hidden layer dimension from 128 to 64 and includes 2 ConvNeXtBlocks; the second stage reduces the dimension from 64 to 32 with 3 ConvNeXtBlocks; the final stage maintains the dimension at 32 with 1 ConvNeXtBlock. Downsampling is achieved by setting the convolutional stride to 2 at the end of the first two stages.

In summary, the **DeepSilencer** network effectively integrates global and local information, demonstrating superior performance in siRNA knockdown efficiency prediction and classification tasks.

## 3 Results

### 3.1 Evaluation on Within-Dataset Cross-Validation

To rigorously evaluate the performance of various models, we conducted five-fold cross-validation on three different datasets: Huesken, Mixset, and Takayuki. For each dataset, we calculated the mean and standard deviation of the Area Under the Receiver Operating Characteristic Curve (AUROC) for each model. The results are summarized in Fig. 3. For detailed numerical results, please refer to Table 2.

The results indicate that DeepSilencer consistently outperformed other models across all datasets. Notably, DeepSilencer demonstrated superior performance on the Mixset and Takayuki datasets, which are relatively smaller in size. This can be attributed to the model’s simple and compact architecture, as well as its use of multi-task supervision, which enhances its ability to generalize well on smaller datasets.

[Figure 3](../assets/figure/figure-3.jpg)

**Fig. 3** : This figure illustrates the performance of various models evaluated using fivefold cross-validation across three datasets: Huesken, Mixset, and Takayuki. Each bar represents the mean Area Under the Receiver Operating Characteristic Curve for a specific model, with error bars indicating the standard deviation. The models compared include OligoWalk, siRNAPred, i-Score, DSIR, OligoFormer, and DeepSilencer. Notably, DeepSilencer demonstrates superior performance across all datasets, achieving the highest AUROC scores, as evident from the consistent pattern of the bars. Detailed numerical results, including mean values and standard deviations, can be found in Table 2.

**Table 2** : AUROC on Within-Dataset Cross-Validation

[Table 2](../assets/table/table-2.csv)

| Methods | Huesken | Mixset | Takayuki |
| --- | --- | --- | --- |
| OligoWalk | 0.781 ± 0.021 | 0.7842 ± 0.035 | 0.792 ± 0.032 |
| siRNAPred | 0.664 ± 0.014 | 0.6158 ± 0.028 | 0.508 ± 0.021 |
| i-Score | 0.831 ± 0.008 | 0.743 ± 0.076 | 0.763 ± 0.035 |
| DSIR | 0.843 ± 0.003 | 0.745 ± 0.068 | 0.772 ± 0.033 |
| OligoFormer | 0.862 ± 0.012 | 0.844 ± 0.025 | 0.863 ± 0.028 |
| DeepSilencer | 0.869 ± 0.010 | 0.879 ± 0.033 | 0.887 ± 0.035 |

### 3.2 Evaluation on Cross-Dataset Validation

To evaluate the robustness and generalizability of our model, we conducted crossdataset validation. Specifically, we trained our model on the Huesken dataset and tested it on a mixed set consisting of siRNA sequences from the Amarzguioui, Haborth, Hsieh, Khvorova, Reynolds, Vickers, and Ui-Tei datasets. The sequences in the mixed set are different from those in the training set, ensuring that the evaluation provides a stringent test of the model’s robustness. We compared the performance of various publicly available models against our DeepSilencer model using the Area Under the Curve (AUC) and F1 score metrics.

As shown in Table 3, our DeepSilencer model demonstrated superior performance across both metrics. Notably, even when using only the siRNA sequence information, DeepSilencer outperformed the previous state-of-the-art (SOTA) model, OligoFormer(siRNA). Furthermore, when incorporating siRNA sequence and thermodynamic features, DeepSilencer surpassed the performance of OligoFormer, which utilizes additional information such as siRNA, mRNA, RNAFM, and TR.

These results underscore the efficacy of our model in accurately predicting siRNA silencing efficiency, highlighting its potential for broader applications in RNA interference research.

**Table 3** : Evaluation on Cross-Dataset Validation. The fields in parentheses represent the features used by the model: siRNA for siRNA sequences, mRNA for mRNA sequences, RNAFM for siRNA sequence embeddings extracted using the RNAFM model, and TR for thermodynamic parameters

[Table 3](../assets/table/table-3.csv)

| Methods | AUC | F1 score |
| --- | --- | --- |
| OligoWalk | 0.708 | 0.6258 |
| siRNAPred | 0.578 | 0.525 |
| i-Score | 0.746 | 0.219 |
| DSIR | 0.748 | 0.693 |
| OligoFormer(siRNA+mRNA+RNAFM+TR) | 0.815 | 0.769 |
| OligoFormer(siRNA) | 0.787 | 0.742 |
| OligoFormer(siRNA+TR) | 0.793 | 0.751 |
| DeepSilencer(siRNA) | 0.802 | 0.760 |
| DeepSilencer(siRNA+TR) | 0.820 | 0.775 |

### 3.3 Ablation Studies

This section presents three ablation studies: _Ablation Study on Various Data Sampling Methods_ , _Ablation Study on Different Types of Tasks_ , and _Ablation Study on Different Model Architectures_ .We use AUROC to evaluate the performance throughout these studies.All the results are evaluated on cross-dataset.

#### Ablation Study on Various Data Sampling Methods

For this study, we evaluated several data sampling methods: standard single-sample training, random pair sampling, Selective Pair Sampling (Easy), Selective Pair Sampling (Hard), and our default method, Selective Pair Sampling. The results are illustrated in Fig. 4a.

Selective Pair Sampling is defined by adjusting the threshold range parameters _α_ 1 and _α_ 2. For Selective Pair Sampling (Easy), we set _α_ 1 = 0.2 and _α_ 2 = 0.4, while for Selective Pair Sampling (Hard), we set _α_ 1 = 0.05 and _α_ 2 = 0.1. In the DeepSilencer model, the default values are α1 = 0.05 and α2 = 0.2. The results of more parameter combinations are presented in Table 4.

Our findings indicate that standard single-sample training (no pair sampling) performed the worst. The performance hierarchy observed was as follows: Selective Pair Sampling (Easy) _<_ random pair sampling _<_ Selective Pair Sampling (Hard), with the adjusted, moderate-difficulty Selective Pair Sampling achieving the best results. This aligns with previous research, which suggests that samples that are too easy or too hard can degrade model performance. Moderately difficult samples provide better gradients, thereby enhancing model learning.

**Table 4** : The impact of different combinations of parameter α1 and α2 on the model.

[Table 4](../assets/table/table-4.csv)

| α1 | α2 | AUROC |
| --- | --- | --- |
| 0.2 | 0.4 | 0.808 |
| 0.05 | 0.1 | 0.815 |
| 0.05 | 0.2 | 0.820 |
| 0.3 | 0.5 | 0.789 |
| 0.2 | 0.5 | 0.804 |
| 0.02 | 0.07 | 0.812 |

#### Ablation Study on Different Types of Tasks

In this study, we compared regression tasks, classification tasks, contrastive tasks, and their combinations. The results are shown in Fig. 4b.

For single tasks, the contrastive task yielded the best model performance. However, when combined into dual tasks, the combination of regression and classification tasks without the contrastive task performed the best. For tri-tasks, we evaluated using the predictions from the regression task and the classification task to calculate the AUROC. Ultimately, we found that using the regression task’s predictions resulted in better performance.

It is evident that using only the classification task resulted in the poorest performance, which was comparable to the level of OligoWalk. However, simply switching from the classification task to the regression task improved the AUROC by 0.15. We attribute this to the fact that knockdown efficiency is inherently a continuous variable rather than a discrete one. For instance, the siRNA sequence **UCCAAAUCAAAGUUAUUUU** has a knockdown efficiency of **0.5212** with a classification label of **0** , while the sequence **AUUCUUCAGCUAGGUCAGC** has a knockdown efficiency of **0.5287** with a classification label of **1** . These two siRNAs should be considered equivalent in terms of their effect, but using classification labels forces a separation between them, which clearly impairs the model’s ability to learn effectively.

#### Ablation Study on Different Model Architectures

The final ablation study focused on different model architectures. We computed both the AUROC and the Area Under the Precision-Recall Curve (AUPRC) to evaluate performance. The results, depicted in Fig. 4c, show that our hybrid model, combining transformer and CNN components, outperformed the others.

These ablation studies provide valuable insights into the optimal configurations for data sampling methods, task types, and model architectures, contributing to the overall improvement of our model’s performance.

[Figure 4](../assets/figure/figure-4.jpg)

**Fig. 4** : Ablation study. (a) Impact of different data sampling methods on model performance. ”No pair sampling” involves random sample selection without contrastive loss. ”Random pair sampling” selects two siRNA samples randomly for contrastive loss. ”Selective Pair Sampling” adjusts α1 and α2 to control knockdown efficiency differences, making comparisons more challenging. ”DeepSilencer” represents our optimal settings. (b) Impact of different tasks on model performance. We evaluated three single tasks (regression, classification, contrastive), three dual-task combinations, and two tri-task settings. ”Tri-task (Discrete)” uses classification predictions, while ”DeepSilencer” uses regression predictions, showing better performance. (c) Results of different model architectures: using only Transformer, only ConvNeXt (with an embedding layer), and the full DeepSilencer model. Evaluations based on AUROC and AUPRC show that DeepSilencer achieved the best performance.

## 4 Discussion

In this study, we introduce DeepSilencer, a model developed to predict the knockdown efficiency of siRNA. A key innovation of DeepSilencer lies in combining Selective Pair Sampling with a multi-task learning framework that incorporates contrastive loss. This combination enhances model performance by effectively distinguishing between siRNA sequences with similar knockdown efficiencies and is particularly beneficial in smaller datasets. Our approach harmoniously integrates different tasks, resulting in significant model improvements (see Fig. 4b). However, it is important to note that as dataset size increases, the relative advantages of these improvements may diminish (see Fig. 4a).

Our research also highlights that focusing solely on the siRNA sequence allows the model to achieve excellent results, showcasing strong generalization capabilities. Additionally, thoughtful model design can further elevate performance (see Fig. 4c). Specifically, DeepSilencer integrates Transformer and ConvNeXt modules to leverage their strengths for enhanced prediction. The Transformer module captures global features of siRNA sequences via its self-attention mechanism, understanding long-range dependencies and interactions across the entire sequence(see Fig. 5a). In contrast, the ConvNeXt module focuses on extracting local features, detecting specific short subsequences or local structural motifs within the siRNA(see Fig. 5b). The combination of these two modules creates a powerful synergy, enabling the model to comprehensively understand siRNA sequences from both overall and local perspectives, which is a key factor in its superior performance.

To further visualize how the model separates siRNAs, we employed the t-SNE dimensionality reduction technique. The distribution of the points in the t-SNE plot reveals several interesting patterns(see Fig. 6). We observe that the points are not randomly scattered but rather form distinct clusters and gradients. The presence of clusters indicates that the DeepSilencer model has learned to group siRNAs with similar characteristics and can differentiate between those with different knockdown efficiencies to some extent. Although there is some overlap between the two classes, the overall trend shows that the model has learned to map siRNAs with higher knockdown efficiency differently from those with lower knockdown efficiency in the reduced-dimensional space. This visualization validates the model’s ability to learn discriminative representations and offers an intuitive understanding of its separation capabilities.

Examining the actual prediction results, as shown in Table 5, we tested the model on the Mixset, which it had not encountered before. Initially, we were concerned that the use of relative position encoding in the transformer and CNN might cause the model to struggle with differentiating between similar sequences, especially those that are shifted versions of each other. However, DeepSilencer effectively distinguished between sequences with high similarity but significantly different knockdown efficiencies.

Focusing on specific samples, we observed that siRNAs in Mixset with high knockdown efficiency are more likely to start with a U and less likely with a G or C, while the last nucleotide is more likely to be a C and less likely an A (see Fig 7). We identified two examples that contradict these statistics: one starting with G and ending with A but with high efficiency, and another starting with U and ending with C but with low efficiency. As expected, the model’s performance on these cases was poor.

From analyzing these examples, we conclude that DeepSilencer has learned general sequence patterns related to siRNA knockdown efficiency from the available data. However, due to data limitations and our evolving understanding of the complex siRNA knockdown process, the model struggles with ”anomalous” siRNAs.

In summary, DeepSilencer proves to be a robust and effective tool for siRNA knockdown prediction. The strategic use of Selective Pair Sampling and contrastive loss, along with a carefully crafted multi-task learning framework and the integration of Transformer and ConvNeXt modules, enables the model to excel, particularly with limited data. These insights contribute valuable advancements to RNA interference research, offering a promising direction for future siRNA prediction models.

[Figure 5](../assets/figure/figure-5.jpg)

**Fig. 5** : Feature Maps. (a) The Transformer module is primarily responsible for capturing global features of siRNA sequences.The feature map illustrates the learned representations across various sequence positions and dimensions. For instance, the brighter yellow regions at specific positions and dimensions imply that the Transformer has recognized these as more significant features within the global context of the siRNA sequence. (b) The ConvNeXt module focuses on extracting local features from the siRNA sequences. The feature map obtained from the ConvNeXt module reveals distinct patterns. The distinct patterns and variations in color intensity highlight the local feature extraction capabilities of ConvNeXt. For example, certain regions with concentrated yellow patches might indicate the presence of specific local motifs that are highly relevant for predicting siRNA knockdown efficiency.

[Figure 6](../assets/figure/figure-6.jpg)

**Fig. 6** : Clustering of latent vectors from the DeepSilencer Model. The x-axis represents the first t-SNE component, and the y-axis represents the second t-SNE component. The data points, colored based on their labels (red for Label 0 and blue for Label 1 corresponding to different knockdown efficiencies), form distinct clusters.

**Table 5** : Selected examples of knockout efficiency prediction on Mixset using the DeepSilencer

[Table 5](../assets/table/table-5.jpg)

*Table retained as an image; cells were not reliably transcribed.*

[Figure 7](../assets/figure/figure-7.jpg)

**Fig. 7** : Nucleotide Distribution at Different Positions for High Knockdown Efficiency siRNAs in Mixset. The chart illustrates the likelihood of each nucleotide (A, U, G, C) appearing at specific positions within siRNAs that exhibit high knockdown efficiency. Notably, the first nucleotide is more frequently U, while G and C are less common. Conversely, the last position shows a higher occurrence of C and a lower occurrence of A, highlighting distinct sequence preferences associated with effective siRNA activity.

## 5 Conclusion

We introduced DeepSilencer, a model for predicting siRNA knockdown efficiency, which demonstrates outstanding performance across multiple datasets, highlighting its robustness and adaptability. This success is attributed to techniques such as Selective Pair Sampling and contrastive loss within a multi-task learning framework.

As larger and more diverse datasets become available, we anticipate that siRNA prediction models will achieve even greater accuracy, which is crucial for developing precise siRNA-based therapies. Moreover, DeepSilencer’s predictive capabilities could significantly aid drug development by optimizing therapeutic candidates and accelerating RNA interference-based treatments. Interestingly, some siRNAs that do not conform to typical high knockdown patterns also exhibit excellent efficiency. We hope future data and research will further enhance our model’s capabilities.

Notably, while the primary focus of this study has been on siRNA, DeepSilencer holds great potential for other RNA-based therapeutics. MicroRNAs (miRNAs) regulate numerous biological processes, short hairpin RNAs (shRNAs) are crucial for RNAi-based strategies, and PIWI-interacting RNAs (piRNAs) are important in transposon silencing [32],[33],[34],[35] and [36]. Given the sequence-based functional similarities among siRNAs, miRNAs, shRNAs, and piRNAs, DeepSilencer’s architecture, which can capture sequence-related features for knockdown prediction, could be adapted to predict miRNA-mediated gene regulation, shRNA-mediated gene silencing and piRNA-related functions like target interactions. This may offer insights into miRNA-related diseases, facilitate miRNA-based therapies like designing better mimics or inhibitors, accelerate shRNA-based gene therapy development for genetic disorders and cancers, and open new research paths for piRNA-associated biology and potential therapies in germline-related diseases.

In summary, DeepSilencer marks a significant advancement in siRNA prediction and opens new avenues for future research and therapeutic advancements in RNA interference. It can accurately rank candidate siRNA sequences using only the sequences themselves and achieves even greater accuracy by incorporating simple thermodynamic parameters. This capability enhances drug development efficiency and reduces costs. Additionally, its potential applications in miRNA and shRNA research broaden its scope and relevance, offering exciting possibilities for further exploration in the field of RNA therapeutics.

## Declarations

### Ethics approval and consent to participate

Not applicable.

### Consent for publication

Not applicable.

### Availability of data and materials

The datasets supporting the conclusions of this article are available in https://github.com/BlackCattt9/DeepSilencer/tree/main/data. The datasets used in this article are sourced from the compiled datasets of oligoformer[17], which you can access from https://github.com/lulab/OligoFormer. These datasets compile nine datasets from the studies by Huesken[18], Takayuki[19], Amarzguioui[20], Haborth[21], Khvorova, Reynolds[22], Vickers[23],UiTei[24] and Hsieh[25].

The code used in this article are available on GitHub and can be accessed at the following link: https://github.com/BlackCattt9/DeepSilencer (we will organize and upload the code).

### Competing interests

The authors declare no competing interests

### Funding

This research is supported by the Research and Development Program of Ministry of Science and Technology of China (ID: 2020AAA0105800).

### Authors’ contributions

WL was primarily responsible for data processing, model construction, coding, computation, and manuscript drafting. WW contributed significantly to the conceptualization of the research idea, analysis and discussion of the results, and manuscript revision. All authors read and approved the final manuscript.

### Acknowledgements

Not applicable.

## References

- [1] Zamore, P.D., Tuschl, T., Sharp, P.A., Bartel, D.P.: Rnai: double-stranded rna directs the atp-dependent cleavage of mrna at 21 to 23 nucleotide intervals. cell **101** (1), 25–33 (2000) https://doi.org/10.1016/S0092-8674(00)80620-0

- [2] Tang, Q., Khvorova, A.: Rnai-based drug design: considerations and future directions. Nature Reviews Drug Discovery **23** (5), 341–364 (2024) https://doi.org/10.1038/s41573-024-00912-9

- [3] Pratt, A.J., MacRae, I.J.: The rna-induced silencing complex: a versatile gene-silencing machine. Journal of Biological Chemistry **284** (27), 17897–17901 (2009) https://doi.org/10.1074/jbc.R900012200

- [4] Deerberg, A., Willkomm, S., Restle, T.: Minimal mechanistic model of sirna-dependent target rna slicing by recombinant human argonaute 2 protein. Proceedings of the National Academy of Sciences **110** (44), 17850–17855 (2013) https://doi.org/10.1073/pnas.121783811

- [5] Lu, Z.J., Mathews, D.H.: Oligowalk: an online sirna design tool utilizing hybridization thermodynamics. Nucleic acids research **36** (suppl_2 ), 104–108 (2008) https://doi.org/10.1093/nar/gkn250

- [6] Han, Y., Liu, Y., Zhang, H., He, F., Shu, C., Dong, L.: Utilizing selected di-and trinucleotides of sirna to predict rnai activity. Computational and Mathematical Methods in Medicine **2017** (1), 5043984 (2017) https://doi.org/10.1155/2017/5043984

- [7] Ichihara, M., Murakumo, Y., Masuda, A., Matsuura, T., Asai, N., Jijiwa, M., Ishida, M., Shinmi, J., Yatsuya, H., Qiao, S., _et al._ : Thermodynamic instability of sirna duplex is a prerequisite for dependable prediction of sirna activities. Nucleic acids research **35** (18), 123 (2007) https://doi.org/10.1093/nar/gkm699

- [8] Lu, Z.J., Mathews, D.H.: Efficient sirna selection using hybridization thermodynamics. Nucleic acids research **36** (2), 640–647 (2008) https://doi.org/10.1093/nar/gkm920

- [9] Lu, Z.J., Mathews, D.H.: Fundamental differences in the equilibrium considerations for sirna and antisense oligodeoxynucleotide design. Nucleic Acids Research **36** (11), 3738–3745 (2008) https://doi.org/10.1093/nar/gkn266

- [10] Han, Y., He, F., Chen, Y., Liu, Y., Yu, H.: Sirna silencing efficacy prediction based on a deep architecture. BMC genomics **19** , 59–65 (2018) https://doi.org/10.1186/s12864-018-5028-8

- [11] La Rosa, M., Fiannaca, A., La Paglia, L., Urso, A.: A graph neural network approach for the analysis of sirna-target biological networks. International Journal of Molecular Sciences **23** (22) (2022) https://doi.org/10.3390/ijms232214211

- [12] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. In: Proceedings of the 31st International Conference on Neural Information Processing Systems. NIPS’17, pp. 6000–6010. Curran Associates Inc., Red Hook, NY, USA (2017). https://doi.org/10.5555/3295222.3295349

- [13] Devlin, J.: Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 (2018) https://doi.org/10.18653/v1/N19-1423

- [14] Xu, J., Xu, N., Xie, W., Zhao, C., Yu, L., Feng, W.: Bert-sirna: sirna target prediction based on bert pre-trained interpretable model. Gene **910** , 148330 (2024) https://doi.org/10.1016/j.gene.2024.148330

- [15] Hochreiter, S.: Long short-term memory. Neural Computation MIT-Press (1997)

- [16] LeCun, Y., Boser, B., Denker, J.S., Henderson, D., Howard, R.E., Hubbard, W., Jackel, L.D.: Backpropagation applied to handwritten zip code recognition. Neural computation **1** (4), 541–551 (1989) https://doi.org/10.1162/neco.1989.1.4.541

- [17] Bai, Y., Zhong, H., Wang, T., Lu, Z.J.: Oligoformer: an accurate and robust prediction method for sirna design. bioRxiv, 2024–02 (2024) https://doi.org/10.1101/2024.02.02.578533

- [18] Huesken, D., Lange, J., Mickanin, C., Weiler, J., Asselbergs, F., Warner, J., Meloon, B., Engel, S., Rosenberg, A., Cohen, D., _et al._ : Design of a genome-wide sirna library using an artificial neural network. Nature biotechnology **23** (8), 995–1001 (2005) https://doi.org/10.1038/nbt1118

- [19] Katoh, T., Suzuki, T.: Specific residues at every third position of sirna shape its efficient rnai activity. Nucleic acids research **35** (4), 27 (2007) https://doi.org/10.1093/nar/gkl1120

- [20] Amarzguioui, M., Holen, T., Babaie, E., Prydz, H.: Tolerance for mutations and chemical modifications in a sirna. Nucleic acids research **31** (2), 589–595 (2003) https://doi.org/10.1093/nar/gkg147

- [21] Harborth, J., Elbashir, S.M., Vandenburgh, K., Manninga, H., Scaringe, S.A., Weber, K., Tuschl, T.: Sequence, chemical, and structural variation of small interfering rnas and short hairpin rnas and the effect on mammalian gene silencing. Antisense and Nucleic Acid Drug Development **13** (2), 83–105 (2003) https://doi.org/10.1089/10872900332162963

- [22] Reynolds, A., Leake, D., Boese, Q., Scaringe, S., Marshall, W.S., Khvorova, A.: Rational sirna design for rna interference. Nature biotechnology **22** (3), 326–330 (2004) https://doi.org/10.1038/nbt936

- [23] Vickers, T.A., Koo, S., Bennett, C.F., Crooke, S.T., Dean, N.M., Baker, B.F.: Efficient reduction of target rnas by small interfering rna and rnase h-dependent antisense agents: a comparative analysis. Journal of Biological Chemistry **278** (9), 7108–7118 (2003) https://doi.org/10.1074/jbc.M210326200

- [24] Ui-Tei, K., Naito, Y., Takahashi, F., Haraguchi, T., Ohki-Hamazaki, H., Juni, A., Ueda, R., Saigo, K.: Guidelines for the selection of highly effective sirna sequences for mammalian and chick rna interference. Nucleic acids research **32** (3), 936–948 (2004) https://doi.org/10.1093/nar/gkh247

- [25] Hsieh, A.C., Bo, R., Manola, J., Vazquez, F., Bare, O., Khvorova, A., Scaringe, S., Sellers, W.R.: A library of sirna duplexes targeting the phosphoinositide 3-kinase pathway: determinants of gene silencing for use in cell-based screens. Nucleic acids research **32** (3), 893–901 (2004) https://doi.org/10.1093/nar/gkh238

- [26] Kihyuk.S: Improved deep metric learning with multi-class n-pair loss objective. In: Advances in Neural Information Processing Systems. NIPS’16, vol. 29, pp. 1857–1865 (2016). https://doi.org/10.5555/3157096.3157304

- [27] Zhang, Y., Yang, Q.: An overview of multi-task learning. National Science Review **5** (1), 30–43 (2018) https://doi.org/10.1093/nsr/nwx105

- [28] Ren, S., He, K., Girshick, R., Sun, J.: Faster r-cnn: Towards real-time object detection with region proposal networks. IEEE Transactions on Pattern Analysis and Machine Intelligence **39** (6), 1137–1149 (2017) https://doi.org/10.1109/TPAMI.2016.2577031

- [29] Woo, S., Debnath, S., Hu, R., Chen, X., Liu, Z., Kweon, I.S., Xie, S.: Convnext v2: Co-designing and scaling convnets with masked autoencoders. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16133–16142 (2023). https://doi.org/10.1109/CVPR52729.2023.01548

- [30] Dai, Z., Liu, H., Le, Q.V., Tan, M.: Coatnet: Marrying convolution and attention for all data sizes. Advances in neural information processing systems **34** , 3965–3977 (2021) https://doi.org/10.5555/3540261.3540564

- [31] Dagli, R.: Astroformer: More data might not be all you need for classification. ArXiv **abs/2304.05350** (2023) https://doi.org/10.48550/arXiv.2304.05350

- [32] Ambros, V.: The functions of animal micrornas. Nature **431** (7006), 350–355 (2004) https://doi.org/10.1038/nature02871

- [33] Jy, Y., Sl, D., Dl, T.: Rna interference by expression of short-interfering rnas and hairpin rnas in mammalian cells. Proceedings of the National Academy of Sciences of the United States of America **99** (9) (2002) https://doi.org/10.1073/pnas.092143499

- [34] Aravin, A., Gaidatzis, D., Pfeffer, S., Lagos-Quintana, M., Landgraf, P., Iovino, N., Morris, P., Brownstein, M.J., Kuramochi-Miyagawa, S., Nakano, T., Chien, M., Russo, J.J., Ju, J., Sheridan, R., Sander, C., Zavolan, M., Tuschl, T.: A novel class of small rnas bind to mili protein in mouse testes. Nature **442** (7099), 203–207 (2006) https://doi.org/10.1038/nature04916

- [35] M, W., P, C., Z, Z., H, L., T, L.: Deepmirtar: A deep-learning approach for predicting human mirna targets. Bioinformatics (Oxford, England) **34** (22) (2018) https://doi.org/10.1093/bioinformatics/bty424

- [36] Khan, S., Khan, M., Iqbal, N., Rahman, M.A.A., Karim, M.K.A.: Deep-pirna: Bi-layered prediction model for piwi-interacting rna using discriminative features. Computers, Materials & Continua **72** (2), 2243–2258 (2022) https://doi.org/10.32604/cmc.2022.022901

## Conversion notes

- Source: arXiv:2503.04200v1, 6 March 2025; https://arxiv.org/html/2503.04200v1. Three associated CSV datasets were retrieved from the authors’ GitHub repository. No separate supplementary document was linked in the supplied paper.
- Dataset CSV columns are retained verbatim: siRNA, mRNA, label, y, td. The CSV files do not provide a formal data dictionary or thermodynamic units; do not infer units or feature ordering from the serialized td column alone.
