<h1 align="center">The Practical Guides for Large Language Models </h1>


<p align="center">
	<img src="https://camo.githubusercontent.com/64f8905651212a80869afbecbf0a9c52a5d1e70beab750dea40a994fa9a9f3c6/68747470733a2f2f617765736f6d652e72652f62616467652e737667" alt="Awesome" data-canonical-src="https://awesome.re/badge.svg" style="max-width: 100%;">	     
</p>

This is an actively updated list of practical guide resources of healthcare LLMs. It's based on our survey paper: [A Survey of Large Language Models for Healthcare: Progress, Application, and Challenge](https://arxiv.org/) and efforts from @[simon599](https://github.com/simon599). 

These sources aim to help practitioners navigate the vast landscape of healthcare-specific large language models (LLMs) and their applications in medical natural language processing (NLP) applications. If you find any resources in our repository helpful, please feel free to use them (don't forget to cite our paper! 😃). 

*image placeholder
<p align="center">
<img width="600" src="./imgs/tree.jpg"/>
</p>

**bibtex to be changed
```bibtex
    @article{yang2023harnessing,
        title={Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond}, 
        author={Jingfeng Yang and Hongye Jin and Ruixiang Tang and Xiaotian Han and Qizhang Feng and Haoming Jiang and Bing Yin and Xia Hu},
        year={2023},
        eprint={2304.13712},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
```

## Latest News💥
- Version 1 live!

**catalog to be updated
## Catalog
* [The Practical Guides for Large Language Models ](#the-practical-guides-for-large-language-models-)
   * [Practical Guide for Models](#practical-guide-for-models)
      * [BERT-style Language Models: Encoder-Decoder or Encoder-only](#bert-style-language-models-encoder-decoder-or-encoder-only)
      * [GPT-style Language Models: Decoder-only](#gpt-style-language-models-decoder-only)
   * [Practical Guide for Data](#practical-guide-for-data)
      * [Pretraining data](#pretraining-data)
      * [Finetuning data](#finetuning-data)
      * [Test data/user data](#test-datauser-data)
   * [Practical Guide for NLP Tasks](#practical-guide-for-nlp-tasks)
      * [Traditional NLU tasks](#traditional-nlu-tasks)
      * [Generation tasks](#generation-tasks)
      * [Knowledge-intensive tasks](#knowledge-intensive-tasks)
      * [Abilities with Scaling](#abilities-with-scaling)
      * [Specific tasks](#specific-tasks)
      * [Real-World ''Tasks''](#real-world-tasks)
      * [Efficiency](#efficiency)
      * [Trustworthiness](#trustworthiness)
      * [Benchmark Instruction Tuning](#benchmark-instruction-tuning)
      * [Alignment](#alignment)
         * [Safety Alignment (Harmless)](#safety-alignment-harmless)
         * [Truthfulness Alignment (Honest)](#truthfulness-alignment-honest)
         * [Practical Guides for Prompting (Helpful)](#practical-guides-for-prompting-helpful)
         * [Alignment Efforts of Open-source Communtity](#alignment-efforts-of-open-source-communtity)
   * [Usage and Restractions (Models and Data)](#Usage-and-Restrictions)

## 2.3 Medical-domain LLMs

### 2.3.1 Pre-training

* BioBERT
  * BioBERT: a pre-trained biomedical language representation model for biomedical text mining. 2020. [paper](https://academic.oup.com/bioinformatics/article-abstract/36/4/1234/5566506)
  * Saama research at mediqa 2019: Pre-trained biobert with attention visualisation for medical natural language inference. 2019. [paper](https://aclanthology.org/W19-5055/)
* PubMedBERT
  * Domain-specific language model pretraining for biomedical natural language processing. 2021. [paper](https://dl.acm.org/doi/abs/10.1145/3458754)
* SciBERT
  * SciBERT: A pretrained language model for scientific text. 2019. [paper](https://arxiv.org/abs/1903.10676)
* ClinicalBERT
  * Publicly available clinical BERT embeddings. 2019. [paper](https://arxiv.org/abs/1904.03323)
* BlueBERT
  * Transfer learning in biomedical natural language processing: an evaluation of BERT and ELMo on ten benchmarking datasets. 2019. [paper](https://arxiv.org/abs/1906.05474)
  * Detecting redundancy in electronic medical records using clinical bert. 2020. [paper](https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/E3-3.pdf)
  * Identification of semantically similar sentences in clinical notes: Iterative intermediate training using multi-task learning. 2020. [paper](https://medinform.jmir.org/2020/11/e22508/)
* BioCPT
  * Biocpt: Contrastive pre-trained transformers with large-scale pubmed search logs for zero-shot biomedical information retrieval. 2023. [paper](https://arxiv.org/abs/2307.00589)
* BioGPT
  * BioGPT: generative pre-trained transformer for biomedical text generation and mining. 2022. [paper](https://academic.oup.com/bib/article-abstract/23/6/bbac409/6713511)
* OphGLM
  * OphGLM: Training an Ophthalmology Large Language-and-Vision Assistant based on Instructions and Dialogue. 2023. [paper](https://arxiv.org/abs/2306.12174)
* GatorTron
  * Gatortron: A large clinical language model to unlock patient information from unstructured electronic health records. 2022. [paper](https://arxiv.org/abs/2203.03540)
  * A large language model for electronic health records. 2022. [paper](https://www.nature.com/articles/s41746-022-00742-2)
* GatorTronGPT
  * A Study of Generative Large Language Model for Medical Research and Healthcare. 2023. [paper](https://arxiv.org/abs/2305.13523)

### 2.3.2 Fine-tuning

* DoctorGLM
  * Doctorglm: Fine-tuning your chinese doctor is not a herculean task. 2023. [paper](https://arxiv.org/abs/2304.01097)
* BianQue
  * BianQue: Balancing the Questioning and Suggestion Ability of Health LLMs with Multi-turn Health Conversations Polished by ChatGPT. 2023. [paper](https://arxiv.org/abs/2310.15896)
* ClinicalGPT
  * ClinicalGPT: Large Language Models Finetuned with Diverse Medical Data and Comprehensive Evaluation. 2023. [paper](https://arxiv.org/abs/2306.09968)
* Qilin-Med
  * Qilin-Med: Multi-stage Knowledge Injection Advanced Medical Large Language Model. 2023. [paper](https://arxiv.org/abs/2310.09089)
* Qilin-Med-VL
  * Qilin-Med-VL: Towards Chinese Large Vision-Language Model for General Healthcare. 2023. [paper](https://arxiv.org/abs/2310.17956)
* ChatDoctor
  * ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge. 2023. [paper](https://www.cureus.com/articles/152858-chatdoctor-a-medical-chat-model-fine-tuned-on-a-large-language-model-meta-ai-llama-using-medical-domain-knowledge.pdf)
* BenTsao
  * Huatuo: Tuning llama model with chinese medical knowledge. 2023. [paper](https://arxiv.org/abs/2304.06975)
* HuatuoGPT
  * HuatuoGPT, towards Taming Language Model to Be a Doctor. 2023. [paper](https://arxiv.org/abs/2305.15075)
* LLaVA-Med
  * Llava-med: Training a large language-and-vision assistant for biomedicine in one day. 2023. [paper](https://arxiv.org/abs/2306.00890)
* Baize-healthcare
  * ?
* Visual Med-Alpeca
  * Visual med-alpaca: A parameter-efficient biomedical
llm with visual capabilities. 2023. [Repo](https://github.com/cambridgeltl/visual-med-alpaca)
* PMC-LLaMA
  * Pmc-llama: Further finetuning llama on medical papers. 2023. [paper](https://arxiv.org/abs/2304.14454)
* Clinical Camel
  * Clinical Camel: An Open-Source Expert-Level Medical Language Model with Dialogue-Based Knowledge Encoding. 2023. [paper](https://arxiv.org/abs/2305.12031)
* MedPaLM 2
  * Towards expert-level medical question answering with large language models. 2023. [paper](https://arxiv.org/abs/2305.09617)
* MedPaLM M
  * Towards generalist biomedical ai. 2023. [paper](https://arxiv.org/abs/2307.14334)

### 2.3.3 Prompting
* DelD-GPT
  * Deid-gpt: Zero-shot medical text de-identification by gpt-4. 2023. [paper](https://arxiv.org/abs/2303.11032)
* ChatCAD
  * Chatcad: Interactive computer-aided diagnosis on medical image using large language models. 2023. [paper](https://arxiv.org/abs/2302.07257)
* Dr. Knows
  * Leveraging a medical knowledge graph into large language models for diagnosis prediction. 2023. [paper](https://arxiv.org/abs/2308.14321)
* MedPaLM
  * Large language models encode clinical knowledge. 2022. [paper](https://arxiv.org/abs/2212.13138)

## 3. Clinical Applications

### 3.1 Medical Diagnosis
* Designing a Deep Learning-Driven Resource-Efficient Diagnostic System for Metastatic Breast Cancer: Reducing Long Delays of Clinical Diagnosis and Improving Patient Survival in Developing Countries. 2023. [paper](https://arxiv.org/abs/2308.02597)
* AI in health and medicine. 2022. [paper](https://www.nature.com/articles/s41591-021-01614-0)
* Large language models in medicine. 2023. [paper](https://www.nature.com/articles/s41591-023-02448-8)
* Leveraging a medical knowledge graph into large language models for diagnosis prediction. 2023. [paper](https://arxiv.org/abs/2308.14321)
* Chatcad: Interactive computer-aided diagnosis on medical image using large language models. 2023. [paper](https://arxiv.org/abs/2302.07257)

### 3.2 Formatting and ICD-Coding
* Applying large language model artificial intelligence for retina International Classification of Diseases (ICD) coding. 2023. [paper](https://www.researchgate.net/profile/Ramesh-Venkatesh/publication/375076384_Applying_large_language_model_artificial_intelligence_for_retina_International_Classification_of_Diseases_ICD_coding/links/653f9d31ff8d8f507cd9ac4c/Applying-large-language-model-artificial-intelligence-for-retina-International-Classification-of-Diseases-ICD-coding.pdf)
* PLM-ICD: automatic ICD coding with pretrained language models. 2022. [paper](https://arxiv.org/abs/2207.05289)
* MIMIC-III, a freely accessible critical care database. 2016. [paper](https://www.nature.com/articles/sdata201635)
* Generative ai text classification using ensemble llm approaches. 2023. [paper](https://arxiv.org/abs/2309.07755)

### 3.3 Clinical Report Generation
* Reporting guidelines for clinical trials evaluating artificial intelligence interventions are needed. 2019. [paper](https://www.nature.com/articles/s41591-019-0603-3)
* Reporting guidelines for clinical trial reports for interventions involving artificial intelligence: the CONSORT-AI extension. 2010. [paper](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(20)30218-1/fulltext)
* Retrieve, reason, and refine: Generating accurate and faithful patient instructions. 2022. [paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/77c08a6e68ae25433f1d117283c0e312-Abstract-Conference.html)
* Chatcad: Interactive computer-aided diagnosis on medical image using large language models. 2023. [paper](https://arxiv.org/abs/2302.07257)
* Can GPT-4V (ision) Serve Medical Applications? Case Studies on GPT-4V for Multimodal Medical Diagnosis. 2023. [paper](https://arxiv.org/abs/2310.09909)
* Qilin-Med-VL: Towards Chinese Large Vision-Language Model for General Healthcare. 2023. [paper](https://arxiv.org/abs/2310.17956)
* Customizing General-Purpose Foundation Models for Medical Report Generation. 2023. [paper](https://arxiv.org/abs/2306.05642)
* Towards generalist foundation model for radiology. 2023. [paper](https://arxiv.org/abs/2308.02463)
* Pmc-vqa: Visual instruction tuning for medical visual question answering. 2023. [paper](https://arxiv.org/abs/2305.10415)
* Clinical Text Summarization: Adapting Large Language Models Can Outperform Human Experts. 2023. [paper](https://arxiv.org/abs/2309.07430)

### 3.4 Medical Education
* Large Language Models in Medical Education: Opportunities, Challenges, and Future Directions. 2023. [paper](https://mededu.jmir.org/2023/1/e48291/)
* Large ai models in health informatics: Applications, challenges, and the future. 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10261199/)
* The Advent of Generative Language Models in Medical Education. 2023. [paper](https://mededu.jmir.org/2023/1/e48163)
* The impending impacts of large language models on medical education. 2023. [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10020064/)

### 3.5 Medical Robotics
* A Nested U-Structure for Instrument Segmentation in Robotic Surgery. 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10218893/)
* The multi-trip autonomous mobile robot scheduling problem with time windows in a stochastic environment at smart hospitals. 2023. [paper](https://www.mdpi.com/2076-3417/13/17/9879)
* Advanced robotics for medical rehabilitation. 2016. [paper](https://link.springer.com/content/pdf/10.1007/978-3-319-19896-5.pdf)
* GRID: Scene-Graph-based Instruction-driven Robotic Task Planning. 2023. [paper](https://arxiv.org/abs/2309.07726)
* Large ai models in health informatics: Applications, challenges, and the future. 2023. [paper](https://ieeexplore.ieee.org/abstract/document/10261199/)
* Trust in Construction AI-Powered Collaborative Robots: A Qualitative Empirical Analysis. 2023. [paper](https://arxiv.org/abs/2308.14846)

### 3.6 Medical Language Translation
* Machine translation of standardised medical terminology using natural language processing: A Scoping Review. 2023. [paper](https://www.sciencedirect.com/science/article/pii/S1871678423000432)
* The Advent of Generative Language Models in Medical Education. 2023. [paper](https://mededu.jmir.org/2023/1/e48163)
* The impending impacts of large language models on medical education. 2023. [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10020064/)

### 3.7 Mental Health Support
* ChatCounselor: A Large Language Models for Mental Health Support. 2023. [paper](https://arxiv.org/abs/2309.15461)
* An overview of the features of chatbots in mental health: A scoping review. 2019. [paper](https://www.sciencedirect.com/science/article/pii/S1386505619307166)
* Tell me, what are you most afraid of? Exploring the Effects of Agent Representation on Information Disclosure in Human-Chatbot Interaction, 2023, [paper](https://link.springer.com/chapter/10.1007/978-3-031-35894-4_13)
* 127
A Brief Wellbeing Training Session Delivered by a Humanoid Social Robot: A Pilot Randomized Controlled Trial
* 56Real conversations with artificial intelligence: A comparison between human–human online conversations and human–chatbot conversations

## 4. Challenges

### 4.1 Hallucination
* 
* 
* 
* 
* 
* 
* 
* 

### 4.2 Lack of Evaluation Benchmarks and Metrics
* 
* 
* 
* 
* 

### 4.3 Domain Data Limitations
* 
* 
* 
* 
* 

### 4.4 New Knowledge Adaptation
* 
* 
* 

### 4.5 Behavior Alignment
* 
* 
* 
* 
* 
* 
* 
* 
* 

### 4.6 Ethical, Legal, and Safety Concerns. 
* 
* 
* 
* 
* 

## 5. Future directions


### 5.1 Introduction of New Benchmarks
* 
* 
* 

### 5.2 Interdisciplinary Collaborations
* 
* 

### 5.3 Multi-modal LLM
* 
* 
* 
* 
* 
* 
* 

### 5.4 LLMs in less established fields of healthcare
* 
* 
* 
* 

***Ignore Below***
### Pretraining data
- **RedPajama**, 2023. [Repo](https://github.com/togethercomputer/RedPajama-Data)
- **The Pile: An 800GB Dataset of Diverse Text for Language Modeling**, Arxiv 2020. [Paper](https://arxiv.org/abs/2101.00027)
- **How does the pre-training objective affect what large language models learn about linguistic properties?**, ACL 2022. [Paper](https://aclanthology.org/2022.acl-short.16/)
- **Scaling laws for neural language models**, 2020. [Paper](https://arxiv.org/abs/2001.08361)
- **Data-centric artificial intelligence: A survey**, 2023. [Paper](https://arxiv.org/abs/2303.10158)
- **How does GPT Obtain its Ability? Tracing Emergent Abilities of Language Models to their Sources**, 2022. [Blog](https://yaofu.notion.site/How-does-GPT-Obtain-its-Ability-Tracing-Emergent-Abilities-of-Language-Models-to-their-Sources-b9a57ac0fcf74f30a1ab9e3e36fa1dc1)
### Finetuning data
- **Benchmarking zero-shot text classification: Datasets, evaluation and entailment approach**, EMNLP 2019. [Paper](https://arxiv.org/abs/1909.00161)
- **Language Models are Few-Shot Learners**, NIPS 2020. [Paper](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html)
- **Does Synthetic Data Generation of LLMs Help Clinical Text Mining?** Arxiv 2023 [Paper](https://arxiv.org/abs/2303.04360)
### Test data/user data
- **Shortcut learning of large language models in natural language understanding: A survey**, Arxiv 2023. [Paper](https://arxiv.org/abs/2208.11857)
- **On the Robustness of ChatGPT: An Adversarial and Out-of-distribution Perspective** Arxiv, 2023. [Paper](https://arxiv.org/abs/2302.12095)
- **SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems** Arxiv 2019. [Paper](https://arxiv.org/abs/1905.00537)





## Practical Guide for NLP Tasks
We build a decision flow for choosing LLMs or fine-tuned models~\protect\footnotemark for user's NLP applications. The decision flow helps users assess whether their downstream NLP applications at hand meet specific conditions and, based on that evaluation, determine whether LLMs or fine-tuned models are the most suitable choice for their applications.
<p align="center">
<img width="500" src="./imgs/decision.png"/>  
</p>

### Traditional NLU tasks

- **A benchmark for toxic comment classification on civil comments dataset** Arxiv 2023 [Paper](https://arxiv.org/abs/2301.11125)
- **Is chatgpt a general-purpose natural language processing task solver?** Arxiv 2023[Paper](https://arxiv.org/abs/2302.06476)
- **Benchmarking large language models for news summarization** Arxiv 2022 [Paper](https://arxiv.org/abs/2301.13848)
### Generation tasks
- **News summarization and evaluation in the era of gpt-3** Arxiv 2022 [Paper](https://arxiv.org/abs/2209.12356)
- **Is chatgpt a good translator? yes with gpt-4 as the engine** Arxiv 2023 [Paper](https://arxiv.org/abs/2301.08745)
- **Multilingual machine translation systems from Microsoft for WMT21 shared task**, WMT2021 [Paper](https://aclanthology.org/2021.wmt-1.54/)
- **Can ChatGPT understand too? a comparative study on chatgpt and fine-tuned bert**, Arxiv 2023, [Paper](https://arxiv.org/pdf/2302.10198.pdf)




### Knowledge-intensive tasks
- **Measuring massive multitask language understanding**, ICLR 2021 [Paper](https://arxiv.org/abs/2009.03300)
- **Beyond the imitation game: Quantifying and extrapolating the capabilities of language models**, Arxiv 2022 [Paper](https://arxiv.org/abs/2206.04615)
- **Inverse scaling prize**, 2022 [Link](https://github.com/inverse-scaling/prize)
- **Atlas: Few-shot Learning with Retrieval Augmented Language Models**, Arxiv 2022 [Paper](https://arxiv.org/abs/2208.03299)
- **Large Language Models Encode Clinical Knowledge**, Arxiv 2022 [Paper](https://arxiv.org/abs/2212.13138)


### Abilities with Scaling

- **Training Compute-Optimal Large Language Models**, NeurIPS 2022 [Paper](https://openreview.net/pdf?id=iBBcRUlOAPR)
- **Scaling Laws for Neural Language Models**, Arxiv 2020 [Paper](https://arxiv.org/abs/2001.08361)
- **Solving math word problems with process- and outcome-based feedback**, Arxiv 2022 [Paper](https://arxiv.org/abs/2211.14275)
- **Chain of thought prompting elicits reasoning in large language models**, NeurIPS 2022 [Paper](https://arxiv.org/abs/2201.11903)
- **Emergent abilities of large language models**, TMLR 2022 [Paper](https://arxiv.org/abs/2206.07682)
- **Inverse scaling can become U-shaped**, Arxiv 2022 [Paper](https://arxiv.org/abs/2211.02011)
- **Towards Reasoning in Large Language Models: A Survey**, Arxiv 2022 [Paper](https://arxiv.org/abs/2212.10403)


### Specific tasks
- **Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks**, Arixv 2022 [Paper](https://arxiv.org/abs/2208.10442)
- **PaLI: A Jointly-Scaled Multilingual Language-Image Model**, Arxiv 2022 [Paper](https://arxiv.org/abs/2209.06794)
- **AugGPT: Leveraging ChatGPT for Text Data Augmentation**, Arxiv 2023 [Paper](https://arxiv.org/abs/2302.13007)
- **Is gpt-3 a good data annotator?**, Arxiv 2022 [Paper](https://arxiv.org/abs/2212.10450)
- **Want To Reduce Labeling Cost? GPT-3 Can Help**, EMNLP findings 2021 [Paper](https://aclanthology.org/2021.findings-emnlp.354/)
- **GPT3Mix: Leveraging Large-scale Language Models for Text Augmentation**, EMNLP findings 2021 [Paper](https://aclanthology.org/2021.findings-emnlp.192/)
- **LLM for Patient-Trial Matching: Privacy-Aware Data Augmentation Towards Better Performance and Generalizability**, Arxiv 2023 [Paper](https://arxiv.org/abs/2303.16756)
- **ChatGPT Outperforms Crowd-Workers for Text-Annotation Tasks**, Arxiv 2023 [Paper](https://arxiv.org/abs/2303.15056)
- **G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment**, Arxiv 2023 [Paper](https://arxiv.org/abs/2303.16634)
- **GPTScore: Evaluate as You Desire**, Arxiv 2023 [Paper](https://arxiv.org/abs/2302.04166)
- **Large Language Models Are State-of-the-Art Evaluators of Translation Quality**, Arxiv 2023 [Paper](https://arxiv.org/abs/2302.14520)
- **Is ChatGPT a Good NLG Evaluator? A Preliminary Study**, Arxiv 2023 [Paper](https://arxiv.org/abs/2303.04048)

### Real-World ''Tasks''
- **Sparks of Artificial General Intelligence: Early experiments with GPT-4**, Arxiv 2023 [Paper](https://arxiv.org/abs/2303.12712)

### Efficiency
1. Cost
- **Openai’s gpt-3 language model: A technical overview**, 2020. [Blog Post](https://lambdalabs.com/blog/demystifying-gpt-3)
- **Measuring the carbon intensity of ai in cloud instances**, FaccT 2022. [Paper](https://dl.acm.org/doi/abs/10.1145/3531146.3533234)
- **In AI, is bigger always better?**, Nature Article 2023. [Article](https://www.nature.com/articles/d41586-023-00641-w)
- **Language Models are Few-Shot Learners**, NeurIPS 2020. [Paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)
- **Pricing**, OpenAI. [Blog Post](https://openai.com/pricing)
2. Latency
- HELM: **Holistic evaluation of language models**, Arxiv 2022. [Paper](https://arxiv.org/abs/2211.09110)
3. Parameter-Efficient Fine-Tuning
- **LoRA: Low-Rank Adaptation of Large Language Models**, Arxiv 2021. [Paper](https://arxiv.org/abs/2106.09685)
- **Prefix-Tuning: Optimizing Continuous Prompts for Generation**, ACL 2021. [Paper](https://aclanthology.org/2021.acl-long.353/)
- **P-Tuning: Prompt Tuning Can Be Comparable to Fine-tuning Across Scales and Tasks**, ACL 2022. [Paper](https://aclanthology.org/2022.acl-short.8/)
- **P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks**, Arxiv 2022. [Paper](https://arxiv.org/abs/2110.07602)
4. Pretraining System
- **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models**, Arxiv 2019. [Paper](https://arxiv.org/abs/1910.02054)
- **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**, Arxiv 2019. [Paper](https://arxiv.org/abs/1910.02054)
- **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM**, Arxiv 2021. [Paper](https://arxiv.org/abs/2104.04473)
- **Reducing Activation Recomputation in Large Transformer Models**, Arxiv 2021. [Paper](https://arxiv.org/abs/2104.04473)


### Trustworthiness
1. Robustness and Calibration
- **Calibrate before use: Improving few-shot performance of language models**, ICML 2021. [Paper](http://proceedings.mlr.press/v139/zhao21c.html)
- **SPeC: A Soft Prompt-Based Calibration on Mitigating Performance Variability in Clinical Notes Summarization**, Arxiv 2023. [Paper](https://arxiv.org/abs/2303.13035)
  
2. Spurious biases
- **Large Language Models Can be Lazy Learners: Analyze Shortcuts in In-Context Learning**, Findings of ACL 2023 [Paper](https://aclanthology.org/2023.findings-acl.284/)
- **Shortcut learning of large language models in natural language understanding: A survey**, 2023 [Paper](https://arxiv.org/abs/2208.11857)
- **Mitigating gender bias in captioning system**, WWW 2020 [Paper](https://dl.acm.org/doi/abs/10.1145/3442381.3449950)
- **Calibrate Before Use: Improving Few-Shot Performance of Language Models**, ICML 2021 [Paper](https://arxiv.org/abs/2102.09690)
- **Shortcut Learning in Deep Neural Networks**, Nature Machine Intelligence 2020 [Paper](https://www.nature.com/articles/s42256-020-00257-z)
- **Do Prompt-Based Models Really Understand the Meaning of Their Prompts?**, NAACL 2022 [Paper](https://aclanthology.org/2022.naacl-main.167/)
  
3. Safety issues
- **GPT-4 System Card**, 2023 [Paper](https://cdn.openai.com/papers/gpt-4-system-card.pdf)
- **The science of detecting llm-generated texts**, Arxiv 2023 [Paper](https://arxiv.org/pdf/2303.07205.pdf)
- **How stereotypes are shared through language: a review and introduction of the aocial categories and stereotypes communication (scsc) framework**, Review of Communication Research, 2019 [Paper](https://research.vu.nl/en/publications/how-stereotypes-are-shared-through-language-a-review-and-introduc)
- **Gender shades: Intersectional accuracy disparities in commercial gender classification**, FaccT 2018 [Paper](https://proceedings.mlr.press/v81/buolamwini18a/buolamwini18a.pdf)


### Benchmark Instruction Tuning

- FLAN: **Finetuned Language Models Are Zero-Shot Learners**, Arxiv 2021 [Paper](https://arxiv.org/abs/2109.01652)
- T0: **Multitask Prompted Training Enables Zero-Shot Task Generalization**, Arxiv 2021 [Paper](https://arxiv.org/abs/2110.08207)
- **Cross-task generalization via natural language crowdsourcing instructions**, ACL 2022 [Paper](https://aclanthology.org/2022.acl-long.244.pdf)
- Tk-INSTRUCT: **Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks**, EMNLP 2022 [Paper](https://aclanthology.org/2022.emnlp-main.340/)
- FLAN-T5/PaLM: **Scaling Instruction-Finetuned Language Models**, Arxiv 2022 [Paper](https://arxiv.org/abs/2210.11416)
- **The Flan Collection: Designing Data and Methods for Effective Instruction Tuning**, Arxiv 2023 [Paper](https://arxiv.org/abs/2301.13688)
- **OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization**, Arxiv 2023 [Paper](https://arxiv.org/abs/2212.12017)

### Alignment

- **Deep Reinforcement Learning from Human Preferences**, NIPS 2017 [Paper](https://arxiv.org/abs/1706.03741)
- **Learning to summarize from human feedback**, Arxiv 2020 [Paper](https://arxiv.org/abs/2009.01325)
- **A General Language Assistant as a Laboratory for Alignment**, Arxiv 2021 [Paper](https://arxiv.org/abs/2112.00861)
- **Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback**, Arxiv 2022 [Paper](https://arxiv.org/abs/2204.05862)
- **Teaching language models to support answers with verified quotes**, Arxiv 2022 [Paper](https://arxiv.org/abs/2203.11147)
- InstructGPT: **Training language models to follow instructions with human feedback**, Arxiv 2022 [Paper](https://arxiv.org/abs/2203.02155)
- **Improving alignment of dialogue agents via targeted human judgements**, Arxiv 2022 [Paper](https://arxiv.org/abs/2209.14375)
- **Scaling Laws for Reward Model Overoptimization**, Arxiv 2022 [Paper](https://arxiv.org/abs/2210.10760)
- Scalable Oversight: **Measuring Progress on Scalable Oversight for Large Language Models**, Arxiv 2022 [Paper](https://arxiv.org/pdf/2211.03540.pdf)

#### Safety Alignment (Harmless)

- **Red Teaming Language Models with Language Models**, Arxiv 2022 [Paper](https://arxiv.org/abs/2202.03286)
- **Constitutional ai: Harmlessness from ai feedback**, Arxiv 2022 [Paper](https://arxiv.org/abs/2212.08073)
- **The Capacity for Moral Self-Correction in Large Language Models**, Arxiv 2023 [Paper](https://arxiv.org/abs/2302.07459)
- **OpenAI: Our approach to AI safety**, 2023 [Blog](https://openai.com/blog/our-approach-to-ai-safety)

#### Truthfulness Alignment (Honest)

- **Reinforcement Learning for Language Models**, 2023 [Blog](https://gist.github.com/yoavg/6bff0fecd65950898eba1bb321cfbd81)

#### Practical Guides for Prompting (Helpful)

- **OpenAI Cookbook**. [Blog](https://github.com/openai/openai-cookbook/blob/main/techniques_to_improve_reliability.md)
- **Prompt Engineering**. [Blog](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
- **ChatGPT Prompt Engineering for Developers!** [Course](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)

#### Alignment Efforts of Open-source Communtity

- **Self-Instruct: Aligning Language Model with Self Generated Instructions**, Arxiv 2022 [Paper](https://arxiv.org/abs/2212.10560)
- **Alpaca**. [Repo](https://github.com/tatsu-lab/stanford_alpaca)
- **Vicuna**. [Repo](https://github.com/lm-sys/FastChat)
- **Dolly**. [Blog](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)
- **DeepSpeed-Chat**. [Blog](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat)
- **GPT4All**. [Repo](https://github.com/nomic-ai/gpt4all)
- **OpenAssitant**. [Repo](https://github.com/LAION-AI/Open-Assistant)
- **ChatGLM**. [Repo](https://github.com/THUDM/ChatGLM-6B)
- **MOSS**. [Repo](https://github.com/OpenLMLab/MOSS)
- **Lamini**. [Repo](https://github.com/lamini-ai/lamini/)/[Blog](https://lamini.ai/blog/introducing-lamini)

## Usage and Restrictions

<!-- We build a decision flow for choosing LLMs or fine-tuned models~\protect\footnotemark for user's NLP applications.  -->
<!-- The decision flow helps users assess whether their downstream NLP applications at hand meet specific conditions and, based on that evaluation, determine whether LLMs or fine-tuned models are the most suitable choice for their applications. -->

We build a table summarizing the LLMs usage restrictions (e.g. for commercial and research purposes). In particular, we provide the information from the models and their pretraining data's perspective.
We urge the users in the community to refer to the licensing information for public models and data and use them in a responsible manner.
We urge the developers to pay special attention to licensing, make them transparent and comprehensive, to prevent any unwanted and unforeseen usage.

<table class="table table-bordered table-hover table-condensed">
    <thead><tr><th title="Field #1">LLMs</th>
    <th title="Field #2" colspan="3" align="center">Model</th>
    <!-- <th title="Field #3"></th> -->
    <!-- <th title="Field #4"></th> -->
    <th title="Field #5" colspan="2" align="center">Data</th>
    <!-- <th title="Field #6"></th> -->
    </tr></thead>
    <tbody><tr>
    <td> </td>
    <td><b>License<b></td>
    <td><b>Commercial Use<b></td>
    <td><b>Other noteable restrictions<b></td>
    <td><b>License<b></td>
    <td><b>Corpus<b></td>
    </tr>
    <tr>
        <td colspan="6" align="left"><b>Encoder-only</b></td>
    <tr>
    <tr>
    <td>BERT series of models (general domain)</td>
    <td>Apache 2.0</td>
    <td>✅</td>
    <td> </td>
    <td>Public</td>
    <td>BooksCorpus, English Wikipedia</td>
    </tr>
    <tr>
    <td>RoBERTa</td>
    <td>MIT license</td>
    <td>✅</td>
    <td> </td>
    <td>Public</td>
    <td>BookCorpus, CC-News, OpenWebText, STORIES</td>
    </tr>
    <tr>
    <td>ERNIE</td>
    <td>Apache 2.0</td>
    <td>✅</td>
    <td> </td>
    <td>Public</td>
    <td>English Wikipedia</td>
    </tr>
    <tr>
    <td>SciBERT</td>
    <td>Apache 2.0</td>
    <td>✅</td>
    <td> </td>
    <td>Public</td>
    <td>BERT corpus, <a href="https://aclanthology.org/N18-3011.pdf">1.14M papers from Semantic Scholar</a></td>
    </tr>
    <tr>
    <td>LegalBERT</td>
    <td>CC BY-SA 4.0</td>
    <td>❌</td>
    <td> </td>
    <td>Public (except data from the <a href="https://case.law/">Case Law Access Project</a>)</td>
    <td>EU legislation,  US court cases, etc.</td>
    </tr>
    <tr>
    <td>BioBERT</td>
    <td>Apache 2.0</td>
    <td>✅</td>
    <td> </td>
    <td><a href="https://www.nlm.nih.gov/databases/download/terms_and_conditions.html">PubMed</a></td>
    <td>PubMed, PMC</td>
    </tr>
    <tr>
        <td colspan="6" align="left"><b>Encoder-Decoder</b></td>
    <tr>
    <tr>
    <td>T5</td>
    <td>Apache 2.0</td>
    <td>✅</td>
    <td> </td>
    <td>Public</td>
    <td>C4</td>
    </tr>
    <tr>
    <td>Flan-T5</td>
    <td>Apache 2.0</td>
    <td>✅</td>
    <td> </td>
    <td>Public</td>
    <td>C4, Mixture of tasks (Fig 2 in paper)</td>
    </tr>
    <tr>
    <td>BART</td>
    <td>Apache 2.0</td>
    <td>✅</td>
    <td> </td>
    <td>Public</td>
    <td>RoBERTa corpus </td>
    </tr>
    <tr>
    <td>GLM</td>
    <td>Apache 2.0</td>
    <td>✅</td>
    <td> </td>
    <td>Public</td>
    <td>BooksCorpus and English Wikipedia</td>
    </tr>
    <tr>
    <td>ChatGLM</td>
    <td><a href="https://github.com/THUDM/ChatGLM-6B/blob/main/MODEL_LICENSE">ChatGLM License</a></td>
    <td>❌</td>
    <td>No use for illegal purposes or military research, no harm the public interest of society</td>
    <td>N/A</td>
    <td>1T tokens of Chinese and English corpus</td>
    </tr>
    <tr>
        <td colspan="6" align="left"><b>Decoder-only</b></td>
    <tr>
    <td>GPT2 </td>
    <td><a href="https://github.com/openai/gpt-2/blob/master/LICENSE">Modified MIT License</a></td>
    <td>✅</td>
    <td>Use GPT-2 responsibly and clearly indicate your content was created using GPT-2.</td>
    <td>Public</td>
    <td>WebText</td>
    </tr>
    <tr>
    <td>GPT-Neo</td>
    <td>MIT license</td>
    <td>✅</td>
    <td> </td>
    <td>Public</td>
    <td><a href="https://pile.eleuther.ai/">Pile</a></td>
    </tr>
    <tr>
    <td>GPT-J</td>
    <td>Apache 2.0</td>
    <td>✅</td>
    <td> </td>
    <td>Public</td>
    <td>Pile</td>
    </tr>
    <tr>
    <td>---&gt; Dolly</td>
    <td>CC BY NC 4.0</td>
    <td>❌</td>
    <td> </td>
    <td>CC BY NC 4.0, Subject to terms of Use of the data generated by OpenAI</td>
    <td>Pile, Self-Instruct</td>
    </tr>
    <tr>
    <td>---&gt; GPT4ALL-J</td>
    <td>Apache 2.0</td>
    <td>✅</td>
    <td> </td>
    <td>Public</td>
    <td><a href="https://huggingface.co/datasets/nomic-ai/gpt4all-j-prompt-generations">GPT4All-J dataset</a></td>
    </tr>
    <tr>
    <td>Pythia</td>
    <td>Apache 2.0</td>
    <td>✅</td>
    <td> </td>
    <td>Public</td>
    <td>Pile</td>
    </tr>
    <tr>
    <td>---&gt; Dolly v2</td>
    <td>MIT license</td>
    <td>✅</td>
    <td> </td>
    <td>Public</td>
    <td>Pile, databricks-dolly-15k</td>
    </tr>
    <tr>
    <td>OPT</td>
    <td><a href="https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/MODEL_LICENSE.md?fbclid=IwAR1BFK5X1XdUpx_QXoiqyfzYWdNAXJPcg8Cf0ddv5T7sa2UrLUvymj1J8G4">OPT-175B LICENSE AGREEMENT</a></td>
    <td>❌</td>
    <td>No development relating to surveillance research and military, no harm the public interest of society</td>
    <td>Public</td>
    <td>RoBERTa corpus, the Pile, PushShift.io Reddit</td>
    </tr>
    <tr>
    <td>---&gt; OPT-IML</td>
    <td><a href="https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/MODEL_LICENSE.md?fbclid=IwAR1BFK5X1XdUpx_QXoiqyfzYWdNAXJPcg8Cf0ddv5T7sa2UrLUvymj1J8G4">OPT-175B LICENSE AGREEMENT</a></td>
    <td>❌</td>
    <td>same to OPT</td>
    <td>Public</td>
    <td>OPT corpus, Extended version of Super-NaturalInstructions</td>
    </tr>
    <tr>
    <td>YaLM</td>
    <td>Apache 2.0</td>
    <td>✅</td>
    <td> </td>
    <td>Unspecified</td>
    <td>Pile, Teams collected Texts in Russian</td>
    </tr>
    <tr>
    <td>BLOOM</td>
    <td><a href="https://bigscience.huggingface.co/blog/the-bigscience-rail-license">The BigScience RAIL License</a></td>
    <td>✅</td>
    <td>No use of generating verifiably false information with the purpose of harming others; <br/>content without expressly disclaiming that the text is machine generated</td>
    <td>Public</td>
    <td>ROOTS corpus (Lauren¸con et al., 2022)</td>
    </tr>
    <tr>
    <td>---&gt; BLOOMZ</td>
    <td><a href="https://bigscience.huggingface.co/blog/the-bigscience-rail-license">The BigScience RAIL License</a></td>
    <td>✅</td>
    <td>same to BLOOM</td>
    <td>Public</td>
    <td>ROOTS corpus, xP3</td>
    </tr>
    <tr>
    <td>Galactica</td>
    <td><a href="https://github.com/paperswithcode/galai/blob/main/LICENSE-MODEL.md">CC BY-NC 4.0</a></td>
    <td>❌</td>
    <td> </td>
    <td>N/A</td>
    <td>The Galactica Corpus</td>
    </tr>
    <tr>
    <td>LLaMA</td>
    <td><a href="https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform">Non-commercial bespoke license</a></td>
    <td>❌</td>
    <td>No development relating to surveillance research and military, no harm the public interest of society</td>
    <td>Public</td>
    <td>CommonCrawl, C4, Github, Wikipedia, etc.</td>
    </tr>
    <tr>
    <td>---&gt; Alpaca</td>
    <td>CC BY NC 4.0</td>
    <td>❌</td>
    <td> </td>
    <td>CC BY NC 4.0, Subject to terms of Use of the data generated by OpenAI</td>
    <td>LLaMA corpus, Self-Instruct</td>
    </tr>
    <tr>
    <td>---&gt; Vicuna</td>
    <td>CC BY NC 4.0</td>
    <td>❌</td>
    <td> </td>
    <td>Subject to terms of Use of the data generated by OpenAI; <br/>Privacy Practices of ShareGPT</td>
    <td>LLaMA corpus, 70K conversations from <a href="http://sharegpt.com/">ShareGPT.com</a></td>
    </tr>
    <tr>
    <td>---&gt; GPT4ALL</td>
    <td>GPL Licensed LLaMa</td>
    <td>❌</td>
    <td> </td>
    <td>Public</td>
    <td><a href="https://huggingface.co/datasets/nomic-ai/gpt4all_prompt_generations">GPT4All dataset</a></td>
    </tr>
    <tr>
    <td>OpenLLaMA</td>
    <td>Apache 2.0</td>
    <td>✅</td>
    <td> </td>
    <td>Public</td>
    <td><a href="https://www.together.xyz/blog/redpajama">RedPajama</a></td>
    </tr>
    <tr>
    <td>CodeGeeX</td>
    <td><a href="https://github.com/THUDM/CodeGeeX/blob/main/MODEL_LICENSE">The CodeGeeX License</a></td>
    <td>❌</td>
    <td>No use for illegal purposes or military research</td>
    <td>Public</td>
    <td>Pile, CodeParrot, etc.</td>
    </tr>
    <tr>
    <td>StarCoder</td>
    <td><a href="https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement">BigCode OpenRAIL-M v1 license</a></td>
    <td>✅</td>
    <td>No use of generating verifiably false information with the purpose of harming others; <br/>content without expressly disclaiming that the text is machine generated</td>
    <td>Public</td>
    <td><a href="https://arxiv.org/pdf/2211.15533.pdf">The Stack</a></td>
    </tr>
    <td>MPT-7B</td>
    <td>Apache 2.0</td>
    <td>✅</td>
    <td> </td>
    <td>Public</td>
    <td><a href="https://arxiv.org/abs/2010.11934">mC4 (english)</a>, <a href="https://arxiv.org/pdf/2211.15533.pdf">The Stack</a>, <a href="https://www.together.xyz/blog/redpajama">RedPajama</a>, <a href="https://aclanthology.org/2020.acl-main.447/">S2ORC</a></td>
    <tr>
        <td><a href="https://huggingface.co/tiiuae/falcon-40b">falcon</a></td>
        <td><a href="https://huggingface.co/tiiuae/falcon-40b/blob/main/LICENSE.txt">TII Falcon LLM License</a></td>
        <td>✅/❌</td>
        <td>Available under a license allowing commercial use</td>
        <td>Public</td>
        <td><a href="https://huggingface.co/datasets/tiiuae/falcon-refinedweb">RefinedWeb</a></td>
    </tr>
    </tbody></table>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Mooler0410/LLMsPracticalGuide&type=Date)](https://star-history.com/#Mooler0410/LLMsPracticalGuide&Date)

