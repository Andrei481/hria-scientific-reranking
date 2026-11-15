---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:4494
- loss:LambdaLoss
base_model: answerdotai/ModernBERT-base
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on answerdotai/ModernBERT-base

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) <!-- at revision 8949b909ec900327062f0ebf497f51aef5e6f0c8 -->
- **Maximum Sequence Length:** 1024 tokens
- **Number of Output Labels:** 1 label
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

### Full Model Architecture

```
CrossEncoder(
  (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'ModernBertForSequenceClassification'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of inputs
pairs = [
    ['In addition, I need to estimate how much these processes cost.', 'Title: Stochastic Generalized Adversarial Label Learning Abstract: The usage of machine learning models has grown substantially and is spreading into several application domains. A common need in using machine learning models is collecting the data required to train these models. In some cases, labeling a massive dataset can be a crippling bottleneck, so there is need to develop models that work when training labels for large amounts of data are not easily obtained. A possible solution is weak supervision, which uses noisy labels that are easily obtained from multiple sources. The challenge is how best to combine these noisy labels and train a model to perform well given a task. In this paper, we propose stochastic generalized adversarial label learning (Stoch-GALL), a framework for training machine learning models that perform well when noisy and possibly correlated labels are provided. Our framework allows users to provide different weak labels and multiple constraints on these labels. Our model then attempts to learn parameters for the data by solving a non-zero sum game optimization. The game is between an adversary that chooses labels for the data and a model that minimizes the error made by the adversarial labels. We test our method on three datasets by training convolutional neural network models that learn to classify image objects with limited access to training labels. Our approach is able to learn even in settings where the weak supervision confounds state-of-the-art weakly supervised learning methods. The results of our experiments demonstrate the applicability of this approach to general classification tasks.'],
    ['In addition, I need to estimate how much these processes cost.', 'Title: Clean or Annotate: How to Spend a Limited Data Collection Budget Abstract: Crowdsourcing platforms are often used to collect datasets for training machine learning models, despite higher levels of inaccurate labeling compared to expert labeling. There are two common strategies to manage the impact of such noise. The first involves aggregating redundant annotations, but comes at the expense of labeling substantially fewer examples. Secondly, prior works have also considered using the entire annotation budget to label as many examples as possible and subsequently apply denoising algorithms to implicitly clean the dataset. We find a middle ground and propose an approach which reserves a fraction of annotations to explicitly clean up highly probable error samples to optimize the annotation process. In particular, we allocate a large portion of the labeling budget to form an initial dataset used to train a model. This model is then used to identify specific examples that appear most likely to be incorrect, which we spend the remaining budget to relabel. Experiments across three model variations and four natural language processing tasks show our approach outperforms or matches both label aggregation and advanced denoising methods designed to handle noisy labels when allocated the same finite annotation budget.'],
    ['In addition, I need to estimate how much these processes cost.', 'Title: Gradient Descent in RKHS with Importance Labeling Abstract: Labeling cost is often expensive and is a fundamental limitation of supervised learning. In this paper, we study importance labeling problem, in which we are given many unlabeled data and select a limited number of data to be labeled from the unlabeled data, and then a learning algorithm is executed on the selected one. We propose a new importance labeling scheme that can effectively select an informative subset of unlabeled data in least squares regression in Reproducing Kernel Hilbert Spaces (RKHS). We analyze the generalization error of gradient descent combined with our labeling scheme and show that the proposed algorithm achieves the optimal rate of convergence in much wider settings and especially gives much better generalization ability in a small label noise setting than the usual uniform sampling scheme. Numerical experiments verify our theoretical findings.'],
    ['In addition, I need to estimate how much these processes cost.', 'Title: Anomaly Detection Under Controlled Sensing Using Actor-Critic Reinforcement Learning Abstract: We consider the problem of detecting anomalies among a given set of processes using their noisy binary sensor measurements. The noiseless sensor measurement corresponding to a normal process is 0, and the measurement is 1 if the process is anomalous. The decision-making algorithm is assumed to have no knowledge of the number of anomalous processes. The algorithm is allowed to choose a subset of the sensors at each time instant until the confidence level on the decision exceeds the desired value. Our objective is to design a sequential sensor selection policy that dynamically determines which processes to observe at each time and when to terminate the detection algorithm. The selection policy is designed such that the anomalous processes are detected with the desired confidence level while incurring minimum cost which comprises the delay in detection and the cost of sensing. We cast this problem as a sequential hypothesis testing problem within the framework of Markov decision processes, and solve it using the actor-critic deep reinforcement learning algorithm. This deep neural network-based algorithm offers a low complexity solution with good detection accuracy. We also study the effect of statistical dependence between the processes on the algorithm performance. Through numerical experiments, we show that our algorithm is able to adapt to any unknown statistical dependence pattern of the processes.'],
    ['In addition, I need to estimate how much these processes cost.', 'Title: Data Programming: Creating Large Training Sets, Quickly Abstract: Large labeled training sets are the critical building blocks of supervised learning methods and are key enablers of deep learning techniques. For some applications, creating labeled training sets is the most time-consuming and expensive part of applying machine learning. We therefore propose a paradigm for the programmatic creation of training sets called data programming in which users express weak supervision strategies or domain heuristics as labeling functions, which are programs that label subsets of the data, but that are noisy and may conflict. We show that by explicitly representing this training set labeling process as a generative model, we can "denoise" the generated training set, and establish theoretically that we can recover the parameters of these generative models in a handful of settings. We then show how to modify a discriminative loss function to make it noise-aware, and demonstrate our method over a range of discriminative models including logistic regression and LSTMs. Experimentally, on the 2014 TAC-KBP Slot Filling challenge, we show that data programming would have led to a new winning score, and also show that applying data programming to an LSTM model leads to a TAC-KBP score almost 6 F1 points over a state-of-the-art LSTM baseline (and into second place in the competition). Additionally, in initial user studies we observed that data programming may be an easier way for non-experts to create machine learning models when training data is limited or unavailable.'],
]
scores = model.predict(pairs)
print(scores)
# [0.8387 0.8056 0.8007 0.7866 0.8081]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'In addition, I need to estimate how much these processes cost.',
    [
        'Title: Stochastic Generalized Adversarial Label Learning Abstract: The usage of machine learning models has grown substantially and is spreading into several application domains. A common need in using machine learning models is collecting the data required to train these models. In some cases, labeling a massive dataset can be a crippling bottleneck, so there is need to develop models that work when training labels for large amounts of data are not easily obtained. A possible solution is weak supervision, which uses noisy labels that are easily obtained from multiple sources. The challenge is how best to combine these noisy labels and train a model to perform well given a task. In this paper, we propose stochastic generalized adversarial label learning (Stoch-GALL), a framework for training machine learning models that perform well when noisy and possibly correlated labels are provided. Our framework allows users to provide different weak labels and multiple constraints on these labels. Our model then attempts to learn parameters for the data by solving a non-zero sum game optimization. The game is between an adversary that chooses labels for the data and a model that minimizes the error made by the adversarial labels. We test our method on three datasets by training convolutional neural network models that learn to classify image objects with limited access to training labels. Our approach is able to learn even in settings where the weak supervision confounds state-of-the-art weakly supervised learning methods. The results of our experiments demonstrate the applicability of this approach to general classification tasks.',
        'Title: Clean or Annotate: How to Spend a Limited Data Collection Budget Abstract: Crowdsourcing platforms are often used to collect datasets for training machine learning models, despite higher levels of inaccurate labeling compared to expert labeling. There are two common strategies to manage the impact of such noise. The first involves aggregating redundant annotations, but comes at the expense of labeling substantially fewer examples. Secondly, prior works have also considered using the entire annotation budget to label as many examples as possible and subsequently apply denoising algorithms to implicitly clean the dataset. We find a middle ground and propose an approach which reserves a fraction of annotations to explicitly clean up highly probable error samples to optimize the annotation process. In particular, we allocate a large portion of the labeling budget to form an initial dataset used to train a model. This model is then used to identify specific examples that appear most likely to be incorrect, which we spend the remaining budget to relabel. Experiments across three model variations and four natural language processing tasks show our approach outperforms or matches both label aggregation and advanced denoising methods designed to handle noisy labels when allocated the same finite annotation budget.',
        'Title: Gradient Descent in RKHS with Importance Labeling Abstract: Labeling cost is often expensive and is a fundamental limitation of supervised learning. In this paper, we study importance labeling problem, in which we are given many unlabeled data and select a limited number of data to be labeled from the unlabeled data, and then a learning algorithm is executed on the selected one. We propose a new importance labeling scheme that can effectively select an informative subset of unlabeled data in least squares regression in Reproducing Kernel Hilbert Spaces (RKHS). We analyze the generalization error of gradient descent combined with our labeling scheme and show that the proposed algorithm achieves the optimal rate of convergence in much wider settings and especially gives much better generalization ability in a small label noise setting than the usual uniform sampling scheme. Numerical experiments verify our theoretical findings.',
        'Title: Anomaly Detection Under Controlled Sensing Using Actor-Critic Reinforcement Learning Abstract: We consider the problem of detecting anomalies among a given set of processes using their noisy binary sensor measurements. The noiseless sensor measurement corresponding to a normal process is 0, and the measurement is 1 if the process is anomalous. The decision-making algorithm is assumed to have no knowledge of the number of anomalous processes. The algorithm is allowed to choose a subset of the sensors at each time instant until the confidence level on the decision exceeds the desired value. Our objective is to design a sequential sensor selection policy that dynamically determines which processes to observe at each time and when to terminate the detection algorithm. The selection policy is designed such that the anomalous processes are detected with the desired confidence level while incurring minimum cost which comprises the delay in detection and the cost of sensing. We cast this problem as a sequential hypothesis testing problem within the framework of Markov decision processes, and solve it using the actor-critic deep reinforcement learning algorithm. This deep neural network-based algorithm offers a low complexity solution with good detection accuracy. We also study the effect of statistical dependence between the processes on the algorithm performance. Through numerical experiments, we show that our algorithm is able to adapt to any unknown statistical dependence pattern of the processes.',
        'Title: Data Programming: Creating Large Training Sets, Quickly Abstract: Large labeled training sets are the critical building blocks of supervised learning methods and are key enablers of deep learning techniques. For some applications, creating labeled training sets is the most time-consuming and expensive part of applying machine learning. We therefore propose a paradigm for the programmatic creation of training sets called data programming in which users express weak supervision strategies or domain heuristics as labeling functions, which are programs that label subsets of the data, but that are noisy and may conflict. We show that by explicitly representing this training set labeling process as a generative model, we can "denoise" the generated training set, and establish theoretically that we can recover the parameters of these generative models in a handful of settings. We then show how to modify a discriminative loss function to make it noise-aware, and demonstrate our method over a range of discriminative models including logistic regression and LSTMs. Experimentally, on the 2014 TAC-KBP Slot Filling challenge, we show that data programming would have led to a new winning score, and also show that applying data programming to an LSTM model leads to a TAC-KBP score almost 6 F1 points over a state-of-the-art LSTM baseline (and into second place in the competition). Additionally, in initial user studies we observed that data programming may be an easier way for non-experts to create machine learning models when training data is limited or unavailable.',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 4,494 training samples
* Columns: <code>query</code>, <code>docs</code>, and <code>labels</code>
* Approximate statistics based on the first 100 samples:
  |          | query                                                                            | docs                                                                                     | labels                                                                                   |
  |:---------|:---------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------|
  | type     | string                                                                           | list                                                                                     | list                                                                                     |
  | modality | text                                                                             |                                                                                          |                                                                                          |
  | details  | <ul><li>min: 8 tokens</li><li>mean: 21.1 tokens</li><li>max: 68 tokens</li></ul> | <ul><li>min: 50 elements</li><li>mean: 50.03 elements</li><li>max: 51 elements</li></ul> | <ul><li>min: 50 elements</li><li>mean: 50.03 elements</li><li>max: 51 elements</li></ul> |
* Samples:
  | query                                                                       | docs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | labels                            |
  |:----------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------|
  | <code>In addition, I need to estimate how much these processes cost.</code> | <code>['Title: Stochastic Generalized Adversarial Label Learning Abstract: The usage of machine learning models has grown substantially and is spreading into several application domains. A common need in using machine learning models is collecting the data required to train these models. In some cases, labeling a massive dataset can be a crippling bottleneck, so there is need to develop models that work when training labels for large amounts of data are not easily obtained. A possible solution is weak supervision, which uses noisy labels that are easily obtained from multiple sources. The challenge is how best to combine these noisy labels and train a model to perform well given a task. In this paper, we propose stochastic generalized adversarial label learning (Stoch-GALL), a framework for training machine learning models that perform well when noisy and possibly correlated labels are provided. Our framework allows users to provide different weak labels and multiple constraints on these labels. Our model then attempts to learn parameters for the data by solving a non-zero sum game optimization. The game is between an adversary that chooses labels for the data and a model that minimizes the error made by the adversarial labels. We test our method on three datasets by training convolutional neural network models that learn to classify image objects with limited access to training labels. Our approach is able to learn even in settings where the weak supervision confounds state-of-the-art weakly supervised learning methods. The results of our experiments demonstrate the applicability of this approach to general classification tasks.', 'Title: Clean or Annotate: How to Spend a Limited Data Collection Budget Abstract: Crowdsourcing platforms are often used to collect datasets for training machine learning models, despite higher levels of inaccurate labeling compared to expert labeling. There are two common strategies to manage the impact of such noise. The first involves aggregating redundant annotations, but comes at the expense of labeling substantially fewer examples. Secondly, prior works have also considered using the entire annotation budget to label as many examples as possible and subsequently apply denoising algorithms to implicitly clean the dataset. We find a middle ground and propose an approach which reserves a fraction of annotations to explicitly clean up highly probable error samples to optimize the annotation process. In particular, we allocate a large portion of the labeling budget to form an initial dataset used to train a model. This model is then used to identify specific examples that appear most likely to be incorrect, which we spend the remaining budget to relabel. Experiments across three model variations and four natural language processing tasks show our approach outperforms or matches both label aggregation and advanced denoising methods designed to handle noisy labels when allocated the same finite annotation budget.', 'Title: Gradient Descent in RKHS with Importance Labeling Abstract: Labeling cost is often expensive and is a fundamental limitation of supervised learning. In this paper, we study importance labeling problem, in which we are given many unlabeled data and select a limited number of data to be labeled from the unlabeled data, and then a learning algorithm is executed on the selected one. We propose a new importance labeling scheme that can effectively select an informative subset of unlabeled data in least squares regression in Reproducing Kernel Hilbert Spaces (RKHS). We analyze the generalization error of gradient descent combined with our labeling scheme and show that the proposed algorithm achieves the optimal rate of convergence in much wider settings and especially gives much better generalization ability in a small label noise setting than the usual uniform sampling scheme. Numerical experiments verify our theoretical findings.', 'Title: Anomaly Detection Under Controlled Sensing Using Actor-Critic Reinforcement Learning Abstract: We consider the problem of detecting anomalies among a given set of processes using their noisy binary sensor measurements. The noiseless sensor measurement corresponding to a normal process is 0, and the measurement is 1 if the process is anomalous. The decision-making algorithm is assumed to have no knowledge of the number of anomalous processes. The algorithm is allowed to choose a subset of the sensors at each time instant until the confidence level on the decision exceeds the desired value. Our objective is to design a sequential sensor selection policy that dynamically determines which processes to observe at each time and when to terminate the detection algorithm. The selection policy is designed such that the anomalous processes are detected with the desired confidence level while incurring minimum cost which comprises the delay in detection and the cost of sensing. We cast this problem as a sequential hypothesis testing problem within the framework of Markov decision processes, and solve it using the actor-critic deep reinforcement learning algorithm. This deep neural network-based algorithm offers a low complexity solution with good detection accuracy. We also study the effect of statistical dependence between the processes on the algorithm performance. Through numerical experiments, we show that our algorithm is able to adapt to any unknown statistical dependence pattern of the processes.', 'Title: Data Programming: Creating Large Training Sets, Quickly Abstract: Large labeled training sets are the critical building blocks of supervised learning methods and are key enablers of deep learning techniques. For some applications, creating labeled training sets is the most time-consuming and expensive part of applying machine learning. We therefore propose a paradigm for the programmatic creation of training sets called data programming in which users express weak supervision strategies or domain heuristics as labeling functions, which are programs that label subsets of the data, but that are noisy and may conflict. We show that by explicitly representing this training set labeling process as a generative model, we can "denoise" the generated training set, and establish theoretically that we can recover the parameters of these generative models in a handful of settings. We then show how to modify a discriminative loss function to make it noise-aware, and demonstrate our method over a range of discriminative models including logistic regression and LSTMs. Experimentally, on the 2014 TAC-KBP Slot Filling challenge, we show that data programming would have led to a new winning score, and also show that applying data programming to an LSTM model leads to a TAC-KBP score almost 6 F1 points over a state-of-the-art LSTM baseline (and into second place in the competition). Additionally, in initial user studies we observed that data programming may be an easier way for non-experts to create machine learning models when training data is limited or unavailable.', ...]</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | <code>[0, 0, 0, 2, 0, ...]</code> |
  | <code>In addition, I need to estimate how much these processes cost.</code> | <code>['Title: Dating Documents using Graph Convolution Networks Abstract: Document date is essential for many important tasks, such as document retrieval, summarization, event detection, etc. While existing approaches for these tasks assume accurate knowledge of the document date, this is not always available, especially for arbitrary documents from the Web. Document Dating is a challenging problem which requires inference over the temporal structure of the document. Prior document dating systems have largely relied on handcrafted features while ignoring such document internal structures. In this paper, we propose NeuralDater, a Graph Convolutional Network (GCN) based document dating approach which jointly exploits syntactic and temporal graph structures of document in a principled way. To the best of our knowledge, this is the first application of deep learning for the problem of document dating. Through extensive experiments on real-world datasets, we find that NeuralDater significantly outperforms state-of-the-art baseline by 19% absolute (45% relative) accuracy points.', 'Title: Data Programming using Continuous and Quality-Guided Labeling Functions Abstract: Scarcity of labeled data is a bottleneck for supervised learning models. A paradigm that has evolved for dealing with this problem is data programming. An existing data programming paradigm allows human supervision to be provided as a set of discrete labeling functions (LF) that output possibly noisy labels to input instances and a generative modelfor consolidating the weak labels. We enhance and generalize this paradigm by supporting functions that output a continuous score (instead of a hard label) that noisily correlates with labels. We show across five applications that continuous LFs are more natural to program and lead to improved recall. We also show that accuracy of existing generative models is unstable with respect to initialization, training epochs, and learning rates. We give control to the data programmer to guide the training process by providing intuitive quality guides with each LF. We propose an elegant method of incorporating these guides into the generative model. Our overall method, called CAGE, makes the data programming paradigm more reliable than other tricks based on initialization, sign-penalties, or soft-accuracy constraints.', 'Title: Online Model Selection: a Rested Bandit Formulation Abstract: Motivated by a natural problem in online model selection with bandit information, we introduce and analyze a best arm identification problem in the rested bandit setting, wherein arm expected losses decrease with the number of times the arm has been played. The shape of the expected loss functions is similar across arms, and is assumed to be available up to unknown parameters that have to be learned on the fly. We define a novel notion of regret for this problem, where we compare to the policy that always plays the arm having the smallest expected loss at the end of the game. We analyze an arm elimination algorithm whose regret vanishes as the time horizon increases. The actual rate of convergence depends in a detailed way on the postulated functional form of the expected losses. Unlike known model selection efforts in the recent bandit literature, our algorithm exploits the specific structure of the problem to learn the unknown parameters of the expected loss function so as to identify the best arm as quickly as possible. We complement our analysis with a lower bound, indicating strengths and limitations of the proposed solution.', 'Title: Categorizing Items with Short and Noisy Descriptions using Ensembled Transferred Embeddings Abstract: Item categorization is a machine learning task which aims at classifying e-commerce items, typically represented by textual attributes, to their most suitable category from a predefined set of categories. An accurate item categorization system is essential for improving both the user experience and the operational processes of the company. In this work, we focus on item categorization settings in which the textual attributes representing items are noisy and short, and labels (i.e., accurate classification of items into categories) are not available. In order to cope with such settings, we propose a novel learning framework, Ensembled Transferred Embeddings (ETE), which relies on two key ideas: 1) labeling a relatively small sample of the target dataset, in a semi-automatic process, and 2) leveraging other datasets from related domains or related tasks that are large-scale and labeled, to extract "transferable embeddings". Evaluation of ETE on a large-scale real-world dataset provided to us by PayPal, shows that it significantly outperforms traditional as well as state-of-the-art item categorization methods.', 'Title: Novel and Efficient Approximations for Zero-One Loss of Linear Classifiers Abstract: The predictive quality of machine learning models is typically measured in terms of their (approximate) expected prediction accuracy or the so-called Area Under the Curve (AUC). Minimizing the reciprocals of these measures are the goals of supervised learning. However, when the models are constructed by the means of empirical risk minimization (ERM), surrogate functions such as the logistic loss or hinge loss are optimized instead. In this work, we show that in the case of linear predictors, the expected error and the expected ranking loss can be effectively approximated by smooth functions whose closed form expressions and those of their first (and second) order derivatives depend on the first and second moments of the data distribution, which can be precomputed. Hence, the complexity of an optimization algorithm applied to these functions does not depend on the size of the training data. These approximation functions are derived under the assumption that the output of the linear classifier for a given data set has an approximately normal distribution. We argue that this assumption is significantly weaker than the Gaussian assumption on the data itself and we support this claim by demonstrating that our new approximation is quite accurate on data sets that are not necessarily Gaussian. We present computational results that show that our proposed approximations and related optimization algorithms can produce linear classifiers with similar or better test accuracy or AUC, than those obtained using state-of-the-art approaches, in a fraction of the time.', ...]</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | <code>[0, 0, 0, 0, 0, ...]</code> |
  | <code>In addition, I need to estimate how much these processes cost.</code> | <code>['Title: Gaussian Process Pseudo-Likelihood Models for Sequence Labeling Abstract: Several machine learning problems arising in natural language processing can be modeled as a sequence labeling problem. We provide Gaussian process models based on pseudo-likelihood approximation to perform sequence labeling. Gaussian processes (GPs) provide a Bayesian approach to learning in a kernel based framework. The pseudo-likelihood model enables one to capture long range dependencies among the output components of the sequence without becoming computationally intractable. We use an efficient variational Gaussian approximation method to perform inference in the proposed model. We also provide an iterative algorithm which can effectively make use of the information from the neighboring labels to perform prediction. The ability to capture long range dependencies makes the proposed approach useful for a wide range of sequence labeling problems. Numerical experiments on some sequence labeling data sets demonstrate the usefulness of the proposed approach.', 'Title: TagRuler: Interactive Tool for Span-Level Data Programming by Demonstration Abstract: Despite rapid developments in the field of machine learning research, collecting high-quality labels for supervised learning remains a bottleneck for many applications. This difficulty is exacerbated by the fact that state-of-the-art models for NLP tasks are becoming deeper and more complex, often increasing the amount of training data required even for fine-tuning. Weak supervision methods, including data programming, address this problem and reduce the cost of label collection by using noisy label sources for supervision. However, until recently, data programming was only accessible to users who knew how to program. To bridge this gap, the Data Programming by Demonstration framework was proposed to facilitate the automatic creation of labeling functions based on a few examples labeled by a domain expert. This framework has proven successful for generating high-accuracy labeling models for document classification. In this work, we extend the DPBD framework to span-level annotation tasks, arguably one of the most time-consuming NLP labeling tasks. We built a novel tool, TagRuler, that makes it easy for annotators to build span-level labeling functions without programming and encourages them to explore trade-offs between different labeling models and active learning strategies. We empirically demonstrated that an annotator could achieve a higher F1 score using the proposed tool compared to manual labeling for different span-level annotation tasks.', "Title: Needmining: Designing Digital Support to Elicit Needs from Social Media Abstract: Today's businesses face a high pressure to innovate in order to succeed in highly competitive markets. Successful innovations, though, typically require the identification and analysis of customer needs. While traditional, established need elicitation methods are time-proven and have demonstrated their capabilities to deliver valuable insights, they lack automation and scalability and, thus, are expensive and time-consuming. In this article, we propose an approach to automatically identify and quantify customer needs by utilizing a novel data source: Users voluntarily and publicly expose information about themselves via social media, as for instance Facebook or Twitter. These posts may contain valuable information about the needs, wants, and demands of their authors. We apply a Design Science Research (DSR) methodology to add design knowledge and artifacts for the digitalization of innovation processes, in particular to provide digital support for the elicitation of customer needs. We want to investigate whether automated, speedy, and scalable need elicitation from social media is feasible. We concentrate on Twitter as a data source and on e-mobility as an application domain. In a first design cycle we conceive, implement and evaluate a method to demonstrate the feasibility of identifying those social media posts that actually express customer needs. In a second cycle, we build on this artifact to additionally quantify the need information elicited, and prove its feasibility. Third, we integrate both developed methods into an end-user software artifact and test usability in an industrial use case. Thus, we add new methods for need elicitation to the body of knowledge, and introduce concrete tooling for innovation management in practice.", "Title: Evaluating Predictive Uncertainty and Robustness to Distributional Shift Using Real World Data Abstract: Most machine learning models operate under the assumption that the training, testing and deployment data is independent and identically distributed (i.i.d.). This assumption doesn't generally hold true in a natural setting. Usually, the deployment data is subject to various types of distributional shifts. The magnitude of a model's performance is proportional to this shift in the distribution of the dataset. Thus it becomes necessary to evaluate a model's uncertainty and robustness to distributional shifts to get a realistic estimate of its expected performance on real-world data. Present methods to evaluate uncertainty and model's robustness are lacking and often fail to paint the full picture. Moreover, most analysis so far has primarily focused on classification tasks. In this paper, we propose more insightful metrics for general regression tasks using the Shifts Weather Prediction Dataset. We also present an evaluation of the baseline methods using these metrics.", 'Title: AI Ethics Statements -- Analysis and lessons learnt from NeurIPS Broader Impact Statements Abstract: Ethics statements have been proposed as a mechanism to increase transparency and promote reflection on the societal impacts of published research. In 2020, the machine learning (ML) conference NeurIPS broke new ground by requiring that all papers include a broader impact statement. This requirement was removed in 2021, in favour of a checklist approach. The 2020 statements therefore provide a unique opportunity to learn from the broader impact experiment: to investigate the benefits and challenges of this and similar governance mechanisms, as well as providing an insight into how ML researchers think about the societal impacts of their own work. Such learning is needed as NeurIPS and other venues continue to question and adapt their policies. To enable this, we have created a dataset containing the impact statements from all NeurIPS 2020 papers, along with additional information such as affiliation type, location and subject area, and a simple visualisation tool for exploration. We also provide an initial quantitative analysis of the dataset, covering representation, engagement, common themes, and willingness to discuss potential harms alongside benefits. We investigate how these vary by geography, affiliation type and subject area. Drawing on these findings, we discuss the potential benefits and negative outcomes of ethics statement requirements, and their possible causes and associated challenges. These lead us to several lessons to be learnt from the 2020 requirement: (i) the importance of creating the right incentives, (ii) the need for clear expectations and guidance, and (iii) the importance of transparency and constructive deliberation. We encourage other researchers to use our dataset to provide additional analysis, to further our understanding of how researchers responded to this requirement, and to investigate the benefits and challenges of this and related mechanisms.', ...]</code> | <code>[0, 1, 0, 0, 0, ...]</code> |
* Loss: [<code>LambdaLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#lambdaloss) with these parameters:
  ```json
  {
      "weighting_scheme": "sentence_transformers.cross_encoder.losses.lambda_loss.NDCGLoss2PPScheme",
      "k": 20,
      "sigma": 1.0,
      "eps": 1e-10,
      "reduction_log": "binary",
      "activation_fn": "torch.nn.modules.linear.Identity",
      "mini_batch_size": 8
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 1
- `num_train_epochs`: 2
- `learning_rate`: 2e-05
- `warmup_steps`: 0.06
- `weight_decay`: 0.01
- `gradient_accumulation_steps`: 8
- `bf16`: True
- `tf32`: False
- `full_determinism`: True
- `seed`: 3407
- `data_seed`: 3407

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 1
- `num_train_epochs`: 2
- `max_steps`: -1
- `learning_rate`: 2e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0.06
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.01
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 8
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1.0
- `label_smoothing_factor`: 0.0
- `bf16`: True
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: True
- `seed`: 3407
- `data_seed`: 3407
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `dataloader_multiprocessing_context`: None
- `dataloader_in_order`: True
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}
- `warmup_ratio`: None

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0018 | 1    | 0.3477        |
| 0.0445 | 25   | 0.6658        |
| 0.0890 | 50   | 0.9349        |
| 0.1335 | 75   | 0.5167        |
| 0.1780 | 100  | 0.4535        |
| 0.2225 | 125  | 0.3642        |
| 0.2670 | 150  | 0.3745        |
| 0.3115 | 175  | 0.5433        |
| 0.3560 | 200  | 0.4564        |
| 0.4005 | 225  | 0.5754        |
| 0.4450 | 250  | 0.3410        |
| 0.4895 | 275  | 0.5858        |
| 0.5340 | 300  | 0.4162        |
| 0.5785 | 325  | 0.4134        |
| 0.6231 | 350  | 0.7212        |
| 0.6676 | 375  | 0.2919        |
| 0.7121 | 400  | 0.3559        |
| 0.7566 | 425  | 0.4324        |
| 0.8011 | 450  | 0.4278        |
| 0.8456 | 475  | 0.4156        |
| 0.8901 | 500  | 0.3428        |
| 0.9346 | 525  | 0.3995        |
| 0.9791 | 550  | 0.3329        |
| 1.0231 | 575  | 0.3979        |
| 1.0676 | 600  | 0.3264        |
| 1.1121 | 625  | 0.3992        |
| 1.1567 | 650  | 0.4145        |
| 1.2012 | 675  | 0.3551        |
| 1.2457 | 700  | 0.3627        |
| 1.2902 | 725  | 0.5832        |
| 1.3347 | 750  | 0.3686        |
| 1.3792 | 775  | 0.3129        |
| 1.4237 | 800  | 0.3936        |
| 1.4682 | 825  | 0.3932        |
| 1.5127 | 850  | 0.3647        |
| 1.5572 | 875  | 0.4347        |
| 1.6017 | 900  | 0.3518        |
| 1.6462 | 925  | 0.4522        |
| 1.6907 | 950  | 0.3665        |
| 1.7352 | 975  | 0.3936        |
| 1.7797 | 1000 | 0.3796        |
| 1.8242 | 1025 | 0.4503        |
| 1.8687 | 1050 | 0.4306        |
| 1.9132 | 1075 | 0.3797        |
| 1.9577 | 1100 | 0.3973        |


### Training Time
- **Training**: 45.0 minutes

### Framework Versions
- Python: 3.11.16
- Sentence Transformers: 6.0.0
- Transformers: 5.16.1
- PyTorch: 2.13.0+cu130
- Accelerate: 1.14.0
- Datasets: 5.0.1
- Tokenizers: 0.23.1

## Additional Resources

- [Training and Finetuning Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-reranker): the end-to-end guide for training or finetuning Cross Encoder (reranker) models.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video reranker models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): training multimodal Cross Encoders.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### LambdaLoss
```bibtex
@inproceedings{wang2018lambdaloss,
  title={The LambdaLoss Framework for Ranking Metric Optimization},
  author={Wang, Xuanhui and Li, Cheng and Golbandi, Nadav and Bendersky, Michael and Najork, Marc},
  booktitle={Proceedings of the 27th ACM international conference on information and knowledge management},
  pages={1313--1322},
  year={2018}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->