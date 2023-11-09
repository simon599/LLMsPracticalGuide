<h1 align="center">The Practical Guides for Medical Large Language Models </h1>




This is an actively updated list of practical guide resources of healthcare LLMs. It's based on our survey paper: [A Survey of Large Language Models for Healthcare: Progress, Application, and Challenge](https://arxiv.org/) and efforts from @[simon599](https://github.com/simon599). 

These sources aim to help practitioners navigate the vast landscape of healthcare-specific large language models (LLMs) and their applications in medical natural language processing (NLP) applications. If you find any resources in our repository helpful, please feel free to use them (don't forget to cite our paper! 😃). 

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
* [Medical-domain LLMs](#23-medical-domain-llms)
   * [Pre-training](#231-pre-training)
   * [Fine-tuning](#232-fine-tuning)
   * [Prompting](#233-prompting)
* [Clinical Applications](#3-clinical-applications)
   * [Medical Diagnosis](#31-medical-diagnosis)
   * [Formatting and ICD Coding](#32-formatting-and-icd-coding)
   * [Clinical Report Generation](#33-clinical-report-generation)
   * [Medical Education](#34-medical-education)
   * [Medical Robotics](#35-medical-robotics)
   * [Medical Language Translation](#36-medical-language-translation)
   * [Mental Health Support](#37-mental-health-support)
* [Challenges](#4-challenges)
   * [Hallucination](#41-hallucination)
   * [Lack of Evaluation Benchmarks and Metrics](#42-lack-of-evaluation-benchmarks-and-metrics)
   * [Domain Data Limitations](#43-domain-data-limitations)
   * [New Knowledge Adaptation](#44-new-knowledge-adaptation)
   * [Behavior Alignment](#45-behavior-alignment)
   * [Ethical, Legal, and Safety Concerns](#46-ethical-legal-and-safety-concerns)
* [Future Directions](#5-future-directions)
   * [Introduction of New Benchmarks](#51-introduction-of-new-benchmarks)
   * [Interdisciplinary Collaborations](#52-interdisciplinary-collaborations)
   * [Multi-modal LLM](#53-multi-modal-llm)
   * [LLMs in Less Established Fields of Healthcare](#54-llms-in-less-established-fields-of-healthcare)


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
* A Brief Wellbeing Training Session Delivered by a Humanoid Social Robot: A Pilot Randomized Controlled Trial. 2023. [paper](https://link.springer.com/article/10.1007/s12369-023-01054-5)
* Real conversations with artificial intelligence: A comparison between human–human online conversations and human–chatbot conversations. 2015. [paper](https://www.sciencedirect.com/science/article/pii/S0747563215001247)

## 4. Challenges

### 4.1 Hallucination
* Survey of hallucination in natural language generation. 2023. [paper](https://dl.acm.org/doi/abs/10.1145/3571730)
* Med-halt: Medical domain hallucination test for large language models. 2023. [paper](https://arxiv.org/abs/2307.15343)
* A survey of hallucination in large foundation models. 2023. [paper](https://arxiv.org/abs/2309.05922)
* Factually Consistent Summarization via Reinforcement Learning with Textual Entailment Feedback. 2023. [paper](https://arxiv.org/abs/2306.00186)
* Improving Factuality of Abstractive Summarization via Contrastive Reward Learning. 2023. [paper](https://arxiv.org/abs/2307.04507)
* Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models. 2023. [paper](https://arxiv.org/abs/2303.08896)
* Retrieval augmentation reduces hallucination in conversation. 2021. [paper](https://arxiv.org/abs/2104.07567)
* Chain-of-verification reduces hallucination in large language models. 2023. [paper](https://arxiv.org/abs/2309.11495)

### 4.2 Lack of Evaluation Benchmarks and Metrics
* What disease does this patient have? a large-scale open domain question answering dataset from medical exams. 2021. [paper](https://www.mdpi.com/2076-3417/11/14/6421)
* Medmcqa: A large-scale multi-subject multi-choice dataset for medical domain question answering. 2022. [paper](https://proceedings.mlr.press/v174/pal22a.html)
* Large language models encode clinical knowledge. 2023. [paper](https://www.nature.com/articles/s41586-023-06291-2)
* Truthfulqa: Measuring how models mimic human falsehoods. 2021. [paper](https://arxiv.org/abs/2109.07958)
* HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models. 2023. [paper](https://ui.adsabs.harvard.edu/abs/2023arXiv230511747L/abstract)

### 4.3 Domain Data Limitations
* Large language models encode clinical knowledge. 2023. [paper](https://www.nature.com/articles/s41586-023-06291-2)
* Towards expert-level medical question answering with large language models. 2023. [paper](https://arxiv.org/abs/2305.09617)
* ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge. 2023. [paper](https://www.cureus.com/articles/152858-chatdoctor-a-medical-chat-model-fine-tuned-on-a-large-language-model-meta-ai-llama-using-medical-domain-knowledge.pdf)
* Textbooks Are All You Need. 2023. [paper](https://arxiv.org/abs/2306.11644)
* Model Dementia: Generated Data Makes Models Forget. 2023. [paper](https://arxiv.org/abs/2305.17493)

### 4.4 New Knowledge Adaptation
* Detecting Edit Failures In Large Language Models: An Improved Specificity Benchmark. 2023. [paper](https://arxiv.org/abs/2305.17553)
* Editing Large Language Models: Problems, Methods, and Opportunities. 2023. [paper](https://arxiv.org/abs/2305.13172)
* Retrieval-augmented generation for knowledge-intensive nlp tasks. 2020. [paper](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)

### 4.5 Behavior Alignment
* Training language models to follow instructions with human feedback. 2022. [paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html)
* Aligning ai with shared human values. 2020. [https://arxiv.org/abs/2008.02275]
* Training a helpful and harmless assistant with reinforcement learning from human feedback. 2022. [paper](https://arxiv.org/abs/2204.05862)
* The power of scale for parameter-efficient prompt tuning. 2021. [paper](https://arxiv.org/abs/2104.08691)
* P-tuning v2: Prompt tuning can be comparable to fine-tuning universally across scales and tasks. 2021. [paper](https://arxiv.org/abs/2110.07602)
* Finetuned language models are zero-shot learners. 2021. [paper](https://arxiv.org/abs/2109.01652)
* Improving alignment of dialogue agents via targeted human judgements. 2022. [paper](https://arxiv.org/abs/2209.14375)
* Webgpt: Browser-assisted question-answering with human feedback. 2021. [paper](https://arxiv.org/abs/2112.09332)
* Languages are rewards: Hindsight finetuning using human feedback. 2023. [paper](https://arxiv.org/abs/2302.02676)

### 4.6 Ethical, Legal, and Safety Concerns
* ChatGPT utility in healthcare education, research, and practice: systematic review on the promising perspectives and valid concerns. 2023. [paper](https://www.mdpi.com/2227-9032/11/6/887)
* ChatGPT listed as author on research papers: many scientists disapprove. 2023. [paper](https://ui.adsabs.harvard.edu/abs/2023Natur.613..620S/abstract)
* A Survey of Large Language Models for Healthcare: from Data, Technology, and Applications to Accountability and Ethics. 2023. [paper](https://arxiv.org/abs/2310.05694)
* Multi-step jailbreaking privacy attacks on chatgpt. 2023. [paper](https://arxiv.org/abs/2304.05197)
* Jailbroken: How does llm safety training fail?. 2023. [paper](https://arxiv.org/abs/2307.02483)

## 5. Future directions


### 5.1 Introduction of New Benchmarks
* A comprehensive benchmark study on biomedical text generation and mining with ChatGPT. 2023. [paper](https://www.biorxiv.org/content/10.1101/2023.04.19.537463.abstract)
* Creation and adoption of large language models in medicine. 2023. [paper](https://jamanetwork.com/journals/jama/article-abstract/2808296)
* Large language models encode clinical knowledge. 2023. [paper](https://www.nature.com/articles/s41586-023-06291-2)

### 5.2 Interdisciplinary Collaborations
* Creation and adoption of large language models in medicine. 2023. [paper](https://jamanetwork.com/journals/jama/article-abstract/2808296)
* ChatGPT and Physicians' Malpractice Risk. 2023. [paper](https://jamanetwork.com/journals/jama-health-forum/fullarticle/2805334)

### 5.3 Multi-modal LLM
* A Survey on Multimodal Large Language Models. 2023. [paper](https://arxiv.org/abs/2306.13549)
* Mm-react: Prompting chatgpt for multimodal reasoning and action. 2023. [paper](https://arxiv.org/abs/2303.11381)
* Minigpt-4: Enhancing vision-language understanding with advanced large language models. 2023. [paper](https://arxiv.org/abs/2304.10592)
* ChatGPT for shaping the future of dentistry: the potential of multi-modal large language model. 2023. [paper](https://www.nature.com/articles/s41368-023-00239-y)
* Frozen Language Model Helps ECG Zero-Shot Learning. 2023. [paper](https://arxiv.org/abs/2303.12311)
* Exploring and Characterizing Large Language Models For Embedded System Development and Debugging. 2023. [paper](https://arxiv.org/abs/2307.03817)
* MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models. 2023. [paper](https://arxiv.org/abs/2306.13394)

### 5.4 LLMs in less established fields of healthcare
* Large language models encode clinical knowledge. 2023. [paper](https://www.nature.com/articles/s41586-023-06291-2)
* Towards expert-level medical question answering with large language models. 2023. [paper](https://arxiv.org/abs/2305.09617)
* A Survey of Large Language Models for Healthcare: from Data, Technology, and Applications to Accountability and Ethics. 2023. [paper]https://arxiv.org/abs/2310.05694()
* Large Language Models in Sport Science & Medicine: Opportunities, Risks and Considerations. 2023. [paper](https://arxiv.org/abs/2305.03851)

Adapted from [https://github.com/Mooler0410/LLMsPracticalGuide](https://github.com/Mooler0410/LLMsPracticalGuide)

