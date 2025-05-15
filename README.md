
#An AI Course Plan

## Prerequisites

- Basic Python programming knowledge

- Basic understanding of machine learning concepts

## Tools and Technologies

- Python 3.x programming 

- NLTK/spaCy

- Jupyter Notebook

- Basic LLM APIs

- HuggingFace library

- Pytorch library 

# Course Tenets 

- Emphasis on practical, hands-on learning

- Focus on understanding fundamentals

- Regular practice exercises

- Real-world applications

- **Using lots of visualization** if implementing code is hard or
  infeasible 

## Weekly Curriculum

## Week 1: Foundations of Language Model

**Lecture (2 hours)**

- Introduction to Language Model

  - Relationship between AI, ML, DL, NLP, LLMs and FMs (Review)

  - NLP in daily interactions (Search Engines, Conversational AI devices
    like Alexa, Siri,.., Spam Filtering, Translation, PoS, Sentiment
    Analysis)

  - What is a model and what a language model is 

  - History of Language Models (Here in timeline we want to show where
    BoW, word2vec and attention come to the LM world. Then we show BERT,
    GPT and so on until now)

    - something like this:

![image](https://github.com/user-attachments/assets/f1f4ddd7-5abf-4590-bc69-de0ad542ea15)


- Different ways to model language (based on the above historical
  picture there are different LMs like Statistical methods, Neural
  Networks, pre-trained and LLMs)

<!-- -->

- Examples of the number of Transformer based models created after
  2019-2024:(https://arxiv.org/pdf/2303.18223) 

![image](https://github.com/user-attachments/assets/2f427bdc-dacd-42e0-891e-acfd29943d97)


- Discriminative vs Generating models

- Language model building blocks:

  - Before diving into the technical details, let\'s get familiar with
    some core concepts in language modeling:

    - [Vocabulary:]{.mark} The set of all possible words that can appear
      in a language model.

    - Context: The sequence of previous words that influence the next
      word.

![image](https://github.com/user-attachments/assets/dfa87652-5800-4a97-8292-c0fed65afdec)


- Probability Distribution: A mathematical function showing the
  likelihood of each possible outcome, given existing sequence.

<!-- -->

- Representing Language 

  - Language entries and NLP Tasks

![image](https://github.com/user-attachments/assets/01aadbeb-9005-4c8d-9d64-cde121bd1899)


- In above picture we should establish the idea of:

  - How to show a word in computer  → Tokenizaing 

  - How to show its meaning in context  \-\--\> Embedding 

<!-- -->

- 

<!-- -->

- [Basic text processing]{.mark} and representation techniques 

  - Data processing for LLMs
  - 
![image](https://github.com/user-attachments/assets/312c6839-2051-4878-a07a-b3b00170b525)


- Pre-processing text 

- From words to numbers

- Tokenization → Embedding 

- Tokenization: → In this section we would like to cover the topics in
  this figure:

![image](https://github.com/user-attachments/assets/cc2d4a95-c133-4f2f-9bcb-f60fee82702e)


- [Different tokenization levels]{.mark} (word, subword, character and
  byte levels)

- Special tokens (Padding, UNK, \...) and why they are needed

- Relation between tokens and vocabulary (how to build vocab out of
  unique tokens in a text)

- Impact of tokenizers on model performance 

<!-- -->

- Tokenizer in action

  - Encoding: converting text to token

  - Decoding: converting token to text

- Tokenizer types

  - [Rule bases]{.mark} (like those that operate on whitespaces between
    words or those that are character based)

  - Leaned tokenizers (BPE for GPT, WordPiece for BERT and Unigram for
    T5) 

<!-- -->

- Using SageMaker JumpStart and HuggingFace

  - Why Model Hub 

  - What is Amazon SageMaker JumpStart

  - What is HuggingFace

  - How to use HuggingFace (pipeline, model and data)

  - Using SageMaker JumpStart and HuggingFace 

  - **Model:** Pre-trained models like BERT or RoBERTa.

  - **Dataset:** Loading and preprocessing data with HuggingFace\'s
    Transformer library.

  - **Pipeline library:** Streamlined workflow combining model,
    tokenizer, and dataset.

![image](https://github.com/user-attachments/assets/b0858d9e-497a-4dbe-867c-221178cde2a5)


**[Lab (2 hours)]{.mark}**

**Environment Setup and Visualization Tools**: Prepares students to use
the \"Tools and Technologies\" listed on page 1, including Python
programming and NLTK

**Basic Text Preprocessing**: This part of the lab covers the
fundamental concept of \"Pre-processing text\". Introduces tokenization,
which is a core concept highlighted in this week.

**Understanding Vocabulary and Frequency Distributions**: This relates
to the \"Vocabulary" concept. Helps visualize word frequencies,
providing insight into the \"Probability Distribution\" concept

**Simple Rule-based Text Representation**: Corresponds to \"Bag of Words
(BoW)\" which is a basic document-level embedding approach. This is a
rule_based approach, and the order is not important here. The key
insight is that BOW discards grammar and word order, focusing only on
whether words occur in the document and how frequently. Creates a basic
matrix representation of text that serves as the foundation for later
weeks

**Tokenization Comparison**: Addresses different tokenization levels
(word, subword, character and byte levels). Exploring \"Tokenizer
types\", specifically the \"Rule bases (like those that operate on
whitespaces between words or those that are character based) .

[NOTE to Lab developers:]{.mark} I deliberately avoided topics that
appeared in later weeks, such as:

- Embeddings (covered in Week 2)

- Contextual embeddings (Week 3)

- Transformers (Week 4)

- Pre-trained models (Week 5)

- Fine-tuning (Weeks 5-11)

- Model architectures (later weeks)

The lab focuses purely on Week 1\'s foundational concepts about how
language is processed at a basic level. This prepares students for the
more complex topics to come while ensuring they have a solid grasp of
the fundamentals.

## Week 2: From  Tokens to Token Embeddings 

**Token Embeddings (1 hour)**

![](media/image8.png){width="5.853210848643919in"
height="5.238531277340332in"}

- Why do we need to tokenize

- From Token (one number)→ to Embedding (many numbers=vector) (review)

- Why we need embeddings (dimensionality reduction, semantic meaning)

- Token-Level Embeddings ( 0.5 hour) 

  - One-Hot Encoding  

    - Basic concept and limitations  

    - Why we need better representations  

  - Advanced Token Embeddings  

    - Word2Vec (CBOW and skip-gram architectures)  

    - GloVe  

    - FastText  

    - Comparison of approaches  

  - Hands-on implementation

-  Document-Level Embeddings (0.5 hour) 

  - Bag of Words (BoW)  

    - Basic concept  

    - Rule-based approach  

    - Order independence  

  - N-gram Models  

    - Extension of BoW  

    - How n affects context capture  

    - Practical considerations  

  - TF-IDF  

    - Theory and implementation  

    - Advantages over BoW  

    - Hands-on practice with each method

- Comparing and Visualization (0.5 hour) 

  - Computing Similarities  

    - Vector normalization  

    - Cosine similarity  

    - When to use each approach  

  - Visualizing Embeddings  

    - Dimensionality reduction techniques  

    - Practical visualization tools  

    - Interpreting embedding spaces  

  - Applications  

    - Retrieval and search

    -   Clustering

    - Data deduplication 

    - Document classification  

    - Recommendation systems

**Lab (1 hour)**

**One hot Encoding**: Implement the basic concept and limitations of
one-hot encoding while introducing the idea of token-level embeddings.

## Week 3:  From Contextual Embedding to Attention: Sequential Modeling 

- Sequence Processing Networks (1 hour) 

  - Examples of sequence problems (time series, translation, sentiment,
    speech generation, image captioning,\...)

  - Why sequential data is different 

  - Why word embedding is not enough ( from embedding → **contextual
    embedding** (one vector) → **attention** that picks the source
    sentence vectors) → Later it was proved that all we needed was just
    that attention part (attention is all you need paper)

![](media/image9.png){width="5.853210848643919in"
height="4.3211001749781275in"}

- RNN Fundamentals  

  - Basic RNN architecture  

  - Processing sequences step by step  

  - RNN Challenges (Vanishing gradients, Long-term dependencies)

  - LSTM/GRU Solutions  

    - Memory cell concept  

    - Gate mechanisms  

- Encoder-Decoder Framework (1 hour) 

  - Types of Sequence Tasks (from diagram)  

    - **NOTE**: Sequential modeling in NLP tasks (use the following
      picture as inspiration to talk about different seq2seq use cases.
      No coding, just introduce the concept of the middle and final
      states and how they lend to different seq2seq NLP
      tasks): <https://karpathy.github.io/2015/05/21/rnn-effectiveness/>

![](media/image10.png){width="5.853210848643919in"
height="1.8990824584426946in"}

-  One-to-One  (classification)

-  One-to-Many  (image captioning)→ One input, sequence output

- Many-to-One  (sentiment analysis) → Sequence input, one output

- Many-to-Many  (translation) → Full sequence in and out : Special case
  of many-to-many that is used in translation. This architecture shows
  the problems we have in long sentence translation and helps us in
  introducing the  attention mechanism.  Encoder: First set of RNN units
  processing input. The final state of the RNN is an encoded version of
  all inputs (hence Encoder here is the first RNN) Decoder: Second set
  of RNN units generating output from the state of the Encoder. Encoder
  processes full input first before decoder comes and use that final
  state.

- Many-to-Many (Simple case of many-to-many ) → PoS (Part of Speech)

<!-- -->

- Understanding Attention Philosophy (45 mins) 

  - The Attention Intuition  

    - Human attention analogy  

![](media/image11.png){width="5.853210848643919in"
height="2.0275218722659667in"}

![](media/image12.png){width="5.853210848643919in"
height="1.6146784776902887in"}

- Review: Why RNN with contextual embedding was not enough

- RNN Attention:

  - Structure:

    - Query (decoder state) looking at Keys (encoder states)  

    - Sequential processing (one token at a time)

    - Attention is between encoder hidden states and current decoder
      state

    - Only decoder attends to encoder outputs

    - Limitations of this approach

<!-- -->

- Bridge to Modern Architectures (1 hour) 

  - High-Level Introduction to Transformers  

![](media/image13.png){width="5.853210848643919in"
height="3.623852799650044in"}

- Encoder-decoder architecture at scale  

- Pre- and Post-Training [concepts]{.underline}

## Week 4: From Transformer Block to Transformer Architecture 

- Evolution to Transformer (30 mins)

  - Why we needed something better than RNN+Attention

  - Review: \"Attention is All You Need\" key insight

  - Transformer Architecture Benefits (Parallel processing, longer
    context, better performance) 

- Transformer Block, A Stack of Technologies  (45 mins) 

![](media/image14.png){width="5.853210848643919in"
height="3.458715004374453in"}

- Inside Embedding block:

  - Tokenizer, embedding the tokens and positional embedding (Review)

![](media/image15.png){width="5.844036526684165in"
height="3.4495406824146984in"}

- Inside Transformer block:

![](media/image16.png){width="5.853210848643919in"
height="2.6055041557305336in"}

- Self-attention

- FFNN

<!-- -->

- Inside Self-attention layer at a glance

  - A simplified view of attention: current position token embedding is
    processed in relation to other token embeddings in the same
    sequence, to get an enriched view of the token (contextual embedding
    of that token)

![](media/image17.png){width="5.853210848643919in"
height="2.8440365266841643in"}

- Step 1: calculate how much each word in that sentence is relevant to
  other words in the same sentence (self). The result is called
  attention score or relevance score. 

- Step 2: multiple each attention score in the meanings of each word in
  that sentence

- Step 3: Average sum to get the attention value of that word/token) in
  that sentence 

![](media/image18.png){width="3.9318143044619425in"
height="4.87471019247594in"}

- Visualizing QKV projection

![](media/image19.png){width="5.853210848643919in"
height="3.7706419510061244in"}

- Attention Head 

  - Each self attention is deemed as one head

![](media/image20.png){width="5.853210848643919in"
height="4.688073053368329in"}

- Multi Head Attention (MHA)

  - Transformers are able to look into each token from different
    perspstives and each needs its own head. They have multiple heads.

![](media/image21.png){width="5.356278433945757in"
height="4.869343832020998in"}

- Inside FFNN Block

  - The output of self attention layer undergoes non-linear
    transformations, which gives the model additional degree of freedom
    to find complex patterns 

  - Bringing the attention weights to the next level

  - Expanding the dimension to original token embedding dimension 

![](media/image22.png){width="5.853210848643919in"
height="4.79816491688539in"}

- Residual connections 

  - What problem they solve 

- Decoder 

  - Difference with Encoder : Casual Attention 

- Encoder-decoder architecture overview

  - Cross-attention

- Visualizing information flows (not mathematical details)

  - From Encoder to Decoder for a Translation scenario

a.  Modern Language Models Architecture (45 mins)

    - Transformer Architecture 

    - Architectural Variations

      - Encoder-only models

      - Decoder-only models 

      - Encode-Decoder models

    - Pre-training and fine-tuning concept

    - When to use each type

      - Sentiment Analysis

      - Semantic Search 

      - Chatbot 

      - Describing what is in the picture 

      - \...(add more examples here)

**Lab (2 hours)**

a.  \...\...

## Week 5: Working with Pre-trained Models 

**Lecture (2 hours)**

- (Review) Model hubs

- Model cards

  - SageMaker JumptStart

  - HuggingFace

- LLM Lifecycle

![](media/image23.png){width="5.853210848643919in"
height="1.715596019247594in"}

- **Pre-training **

  - Common data sets

  - Default Training objectives

![](media/image24.png){width="5.853210848643919in"
height="3.5963298337707785in"}

- MLM (→ BERT)

- NSP (→ BERT)

- Causal Language Modeling (CLM)  (→ GPT)

- What if none of the above objectives fits the current use case?

  - Partial Fine tuning (Transfer Learning)

  - Full Fine tuning

  - Building model from scratch 

<!-- -->

- **Post Training:** Instruction Fine Tuning 

  - High quality data: pairs of (Prompt, Response)

  - Fine tuning entire model

  - Transfer Learning (Partial training)

- **Post Training:** Preference Tuning (Post Training) 

  - Why do we need Preference Tuning 

  - RLHF

  - DPO (Direct Preference Optimization) 

<!-- -->

- Transformer Architecture variations

![](media/image25.png){width="5.853210848643919in"
height="3.5321095800524933in"}

- Encoder variations (Autoencoding LMs)

  - Understanding Language task 

  - BERT

  - RoBERTa

  - DistilBERT

- Decoder variations (Autoregressive LMs)

  - Generating task 

  - GPT

  - LLaMA

- Encoder-Decoder (Combination)

  - T5

  - BART

  - ByT5

<!-- -->

- Scaling Laws

  - Estimating Memory Requirement of a model

    - Using back of napkin method

  - Power Low

  - Chinchilla

![](media/image26.png){width="5.853210848643919in"
height="3.2752285651793525in"}

- Bigger models \# Better Performance 

<!-- -->

- Hyper-parameters of Transformers

  - Vocabulary size

  - Input size 

  - Token embedding dimension

  - FF network embedding size

  - Attention Layer embedding 

  - Number of attention heads

  - Number of stacked layers

- Profiling pre-trained models (like GPT2, Qwen, Small LM)- Model
  Selection

  - Understanding model capabilities and limitations

  - Model selection criteria 

    - Workflow 

![](media/image27.png){width="5.853210848643919in"
height="2.733944663167104in"}

![](media/image28.png){width="5.853210848643919in"
height="1.990825678040245in"}

-  Hosting Model API

![](media/image29.png){width="5.853210848643919in"
height="3.394494750656168in"}

- Why hosting Model API 

- Model capability vs evaluation 

- Taxonomy of Evaluation metrics

![](media/image30.png){width="5.853210848643919in"
height="2.8165135608048995in"}

-  Understanding Public Benchmarks

-  Introduction to eleuther.ai

-  Introduction to Public Leaderboards 

- Introduction to HuggingFace Leaderboard 

![](media/image31.png){width="2.725in" height="1.8in"}

-  Designing an Evaluation Pipeline

- 

**Lab (2 hours)**

- \.....

- 

## Week 6:  Text Classification with Fined-Tuned Encoder Models

![](media/image32.png){width="5.853210848643919in"
height="5.348623140857393in"}

**Fine-Tuned Encoder Models (Task Specific Models) (\... hour)**

- Text classification fundamentals

- Loading data

- Model Leaderboard:

  - [Open LLM
    Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/)

  - Massive Text Embedding Benchmark (MTEB)

![](media/image33.png){width="5.853210848643919in"
height="2.990825678040245in"}

- <https://artificialanalysis.ai/>

- BERT variations and comparing performance benchmarks

![](media/image34.png){width="5.853210848643919in"
height="1.7431189851268591in"}

**Lab (\... hours)**

..

- 

## Week 7:  Expanding Pre-trained Embedding Model 

![](media/image35.png){width="5.853210848643919in"
height="3.376146106736658in"}

**Lecture (2 hours)**

- Sentiment Analysis by adding a Classifier head to an Embedding Model
  with labeled data

  - Data Preparation

  - Generating Sentence Embeddings 

  - Training a Classifier with embedded sentences 

![](media/image36.png){width="5.853210848643919in"
height="2.7981649168853893in"}

- Sentiment Analysis using Embedding Model with unlabeled data

  - Using Zero-Shot classification technique 

  - Using Cosine Similarity to find sentiment

**Lab (2 hours)**

- \...

- Implementing efficient retrieval

- Developing ranking systems

- Search quality assessment

- System optimization techniques

## Week 8: Application of Embedding Models in Information Retrieval  (IR)

**Lecture (2 hours)**

- IR in RAG

![](media/image37.png){width="3.2110597112860892in"
height="1.7014774715660543in"}

- Comparing lexical search vs. Semantic Search (Keyword matching  vs.
  Dense Retrieval) 

- Vector and Vector Data Store

  - Application of KNN in IR

  - KNN libraries

  - Indexing and Vector Databases 

- Building an Semantic Search pipeline through Embedding models

![](media/image38.png){width="5.853210848643919in"
height="2.137613735783027in"}

- Text pre-processing 

- Selecting and applying a chunking strategy

- Sentence Embedding 

- Index Building

- Running Similarity Search 

<!-- -->

- 

- Fine tuning embedding model for Semantic search 

  - Fine tuning chunk embeddings 

![](media/image39.png){width="5.853210848643919in"
height="2.697247375328084in"}

**Lab (2 hours)**

- \...\...

- Implementing efficient retrieval

- Developing ranking systems

- Search quality assessment

- System optimization techniques

## Week 9: Re-ranking in RAG systems

![](media/image40.png){width="5.853210848643919in"
height="2.577981189851269in"}

**Lecture (2 hours)**

- Limitations of pure vector search in RAG systems

- Why ranking matters in retrieval

- Two-stage retrieval architecture

- Understanding re-ranking models (like BGE-reranker-large)

- RAGAS evaluation metrics

  - Answer Relevancy

  - Answer Similarity

  - Context Relevancy

  - Answer Correctness

**Lab (1.5hours)**

**\...**

## Week 10: Using Generative Models for Sentiment Analysis 

**Lecture (2 hours)**

- Challenge of using Generative models for task specific use cases

- Understanding the significance of Context and Prompt

- Using T5 models for Classification 

  - T5 Architecture 

  - Token spans

  - Flan-T5 model family 

    - Multi-task instruction dataset finetuning 

    - How to minimize the effect of "Catastrophic Forgetting"

![](media/image41.png){width="5.853210848643919in"
height="2.724770341207349in"}

- Loading model 

- preparing data (text and label) → Converting a dataset to instruction
  dataset

- Using instruction dataset to generate the required output

- Evaluating model

- 

**Lab (2 hours)**

- in addition to the comment I provided, implementing the following
  environment seems a good candidate as it has *Sentiment Analysis*:

### Analyze customer reviews using Amazon Bedrock

<https://aws.amazon.com/blogs/machine-learning/analyze-customer-reviews-using-amazon-bedrock/>

![](media/image42.png){width="5.853210848643919in"
height="3.100916447944007in"}

- Lab:

## Week 11: Controlling and enhancing the generation process

- Why controlling the output is important

  - Inconsistency 

  - Hallucination 

  - Formatted output 

- Controlling the outputs of Generative Models

  - Token Generation methods

    - Deterministic methods

      - Greedy Search

      - Beam Search

    - Probabilistic methods (Sampling Strategies)

      -  Sampling-based generation

      - Temperature sampling

      - Top p nucleus sampling

      - Top K

  - Experimentation with Modular prompt 

    - Instruction

    - Data

    - Output indicators

    - \...\...\...( add more modular components of complex prompt )

  - Understanding underlying prompt specific model

  - Prompt Chaining

    - Complex multi step task 

![](media/image43.png){width="3.885439632545932in"
height="5.164347112860892in"}

- Reasoning by CoT

  - Reasoning: Self Reflection

  - Variations

    - CoT with one shot

    - CoT with zero shot

- Self Consistency and CoT

  - Using Same prompt with different LLMs

![](media/image44.png){width="3.824779090113736in"
height="5.161653543307087in"}

- Controlling outputs

  - Different types of output verifications

  - Example: LMQL

<!-- -->

- Fine Tuning Generating models

![](media/image45.png){width="5.853210848643919in"
height="1.9541283902012248in"}

- 

- Fine tuning pipeline 

  - Pertaining a Language Model

  - Supervised fine-tuning (SFT)

    - Full fine tuning 

    - Benefits and disadvantages of full fine tuning 

    - Parameter Efficient Fine Tuning (PEFT)

  - Preference tuning (Aligning/RLHF)

<!-- -->

- Selecting evaluation metrics

  -  BLEU

  - ROUGE

- Public Benchmarks

  - GLEU

  - MMLU

  - TruhfulQA

  - GSM8k

**Lab **
course.md…]()

# Diagrams 



[Evaluation Taxonomy](https://viewer.diagrams.net/?tags=%7B%7D&highlight=0000ff&edit=_blank&layers=1&nav=1&title=AI%20Concepts.drawio#R7V1rc5s8Fv41nmk%2FpMP98jEXp%2B1uss026XS7X3ZkkG0aQC7g2O6vXwmQjS4JODbYiem888aIu55zec7RkRjol9HycwJm01vkw3CgKf5yoF8NNE1VdAv%2FIS2rosVSyoZJEvhFk7JpuA%2F%2BQnpm2ToPfJiWbUVThlCYBTO20UNxDL2MaQNJghbsYWMU%2BkzDDEyg0HDvgVBs%2FRn42bRs1RRls%2BMLDCZTemuXvk8E6NFlQzoFPlpUmvThQL9MEMqKX9HyEoak99iOuX5m7%2FrJEhhnTU4Y%2FsuLbn4pmfX78uwbgP%2F8NlHHZ%2BVVnkA4L9%2B4fNhsRbtgkqD5TLwZPRMmGVzKsAAjeoXN22I5gSiCWbLCx5VnlXcsBUS17WJ7UeluWy0Pmla62tHLI0GJ8WR96U0v4B9lR8g75U%2F6v9vHhf3v%2Fwaprf8ezhfq%2BJekU4bkN8gCFA80C0SzgX4Rj1LyB2%2BH%2BIEu%2FOAJ%2F5yQnw9giWIUregu%2FAyVvUL%2F4u4j7dMswg95peKfIAwmMf7t4c6GCW4gnRxgoTwvd0SB75PTLxKYBn%2FLjlbw9gwFcZZ3h3kxMK%2FIteYZSgu1IpdOswQ9wksUInzdqxjF5CrjIAy5JhHvF8WHFwIR7BJezdYEeB1FBNdQWsJWqxd4rKgz8tObJ%2BHqIgHeIyQALaZBBu9nwCP7FtjcsaBhLYl96JdbIRjB8A73fC4z%2BlUIx1kFyBtu9xpQCn1SdMXzyO8doCULxlodBbg0CVym3hZeuoDXA0gfyZ0mMUpx3xBTi%2B8UeOmJatba13LIOaIdNSTQtYac0Wvay36P9nyNblltAWSKbi7GQjwjz3hG%2F639Xem85N4vnYGYgdb6Mye85oKo4FnZ1eeEG1DF2xxCHeMtBOk8gfigeezhXgRY4cijBOTKUcksZwn0A690xNS7Fnen7lXwx5hjpdg4KNzb8W%2BmkOdL4DiIsWzlQKb5bRQ0xv%2BD9OTLnOFEWB6CeELo6BRunis%2Fc5agERgFYZA%2Fvh9gyxCM5sUzY7Ax1vSsLJmTP8QwPNOxPHcQXu4OJrMQLot7ySFTvmDaie9ciAQQezNdP1BMLCah2I8QP%2Bz1AiX%2BJ7x9gxYYNHx89WbplLBrTRnBLCv2rrFJm75OY4PNafXbN9eayZprXbAFlt6lMbAFY%2FDhYQpTmMdRcCMztew3IAoeg7AQlzFKIoDV%2BePeJeKALnyM4uy%2BfEZJGLYnf66yAmKI3kLqz1tjzs5zTOx%2BBr1gnDOxTZx04mzM5ciYaYroWV2yMbdnY1vZ47X61bAzVWsLMVUVEIL%2BBFLDg5JsiiYIW9rhprWCBlGTzTE3CM1KiH5jf70qM21Ea1g4t%2BvfFM0TD770EmXvkCd%2FEYYEhthuPLFpN1mflqfeEQuwgc%2BwOfg0k71EBpIJzMqzOGTWj7EDWIqgX5chyt0nzAEhb5jOsF2EEnJ0ypZS45GTKJ5qdmkqVf09KJ7WjeJZh1Y8MaF3EzwSvfNCkKaEl5R52zwuC2H0qde%2Fqv4ZllGvf3an%2Bidmjr4SiQtItAlBusojVPK7YJuwCF9zY1sxssooPzABfs428ojk9mZQSRf2YlDRY0uvFQO9U8aqWu%2FBDJsd8R8ub78eVuvKDIvJxG8zkkPq6c929EdTJOOf3dIf5z3onX0Y%2BtO53ol5u5L%2BPBSp3AmMYVIyoFPWM8NtoGeyjGt7etanZGooCT8WbWkCZN2mZLT3kJLROkrJqIrB4ed0GxpqYk7m62aoi%2FKRPCCAlTEwMoBRcpbxPPaK2IEMeenXPXVh9JNPctuSlKnbpUmlFrQ6RjEFWQ4xiDcQ0%2FHPBUoeSes4QZEAN75MABeztXSeLM6iIkuA1mWWuD2g30NsSMd82zbEDje0gUlrt3ZYrOG6D6IgBEmhZlFR%2BCErGuitbcXaqiyKpsiGuiWwmhjz%2FyyKSHLnmafopuCJNH3Ode8hmefFLx8SOIa4fzyyywcZ%2BCipI5Bk%2BAZMScGgrMyhxhxMQBCnWa55%2BAWneXnO2y0yaEuOBGtgiIGQ1Ji3V5IrVhYIEJ10IIQDGovFTBK8yiBrUffF4PXLPMqLAMsaECzLvf1mQNQbZCA6ZVG6GCANl8AjJvQWZN60B5ADkB0ikVnOTsMdXZyocgOXpLdx4z0ECYHwtCGjHIVC5khIk9IlZsZ7iFz0shdbTyEZdLrQugiy29CFvmg1dIHYzeXzTnjjuFY5gct%2BGEYj6PtFwfgm9vnY81OpmWUDHd2R%2BMlOK191MdDpCSqDmc4PP%2BvNCGp7qXo68biC2flsliDgTfshaA49Q6kPCbslNuKoZq9wLGS8wikiselY4cQo%2FhJ3dlZJw0hmMo0SvqV2xtP1%2FO%2FfFQ1RCpf6YegHhDZcBWm21WyTfT5XfDZJQNTPdBJl1TU4WZU4dJl9cVuTVTF7cdvPIZZ4Al3ixzstAjTEFEXvCVjI%2BImEWjPq1dpMQkPMSlzcDH9ILOb3bz8%2BDxvb19vhw%2FDb94F2kf%2FX8KRP%2BF8fZb1SbGQ22W7NJhsCIm8wMULXKMrTEi8d2FFdsKrzdW9utxkUw9wJVvV1sJLZ%2Bqv%2FELH4tN78lW%2BadPNqWUpNsbWqbt3BJMAvTnQ%2Bb8Se5WktZXsWGZqhrBUZOky2Z5E5TxKwqhxQmq7nJUrlQ4%2BynOC66Ql66cM3klY8w37l7g1UMddCbrgdWQmLJjrXeVajkZUQr2Q6NVcqFKI9eyOmDi6x4yUrbCjVqpPTdv5c1bruitmD9QSgbnh%2BXwxdg5mlsYql24aAWbfLB0l4%2FvD7w72HErhVZqXK8dETZC4gJ%2Fg9k99SMGRMvj3B2I3yHQeTp9V1rRdnKix6hs1domV%2BTt%2BzmgorCuo4CFmAaoxuW8Ju8uPBNNlQV0PTnrAfJGzdxDfbRTf7VhK9IZHdvzbtBppYkfwjJWuYPcM45lF47pGKtueZwghlGYokniVDnG6geRZiPnq5XrF4e2C2mPams5WDMoXpdJ0xU%2BTn32E5Eed43PhA0y3PgaMx58txuw%2BgM%2FZahMxRORtnWgJkna7HYO4WSu%2BewjFtJolzpnxSdP3Y8jhUsA9gC1%2BVx3EMjjaWfP65yN6ilkN%2BfDtZHFMMC%2B8SFM3Ic4HYJ73IzRCoXZ%2Fwz5xO78pnD1SmAebrhQRZCsNxH2jUrxBCBaB2oZjW6rKok3rTgYaltmIOBP21XYeBz9K6HQiwxHQB%2FVyEOKv6OIINi%2B0x021YsN8aeaIT4E8y2LCaBhv716jdQDuhYIOlFDKF6TTYoEubVbr%2Bc7EeCpmhWEsVvq%2B5wbGzgYPGKwrLCyxJeVenn%2FiwZCEmO%2BH0BYej1jscrofHjgc9T4bJyDFJnUWrU%2FOd2s6XJX%2Fb6%2FyDjLum2Clk5%2BR7T0Tb8oUfPdp8HYRtUTejqUfa%2B2pUu2EkRlX%2FmJNH05Rb%2BSrv%2B9SW8ViTa4tvjSzTalFbXI1VFqMpoWtNW%2BzdIpjdUytFhzOpFfvoSmSovNbrmXFUekYHK6r0A980prkKujgbsz6ihIcUmY7qogmbgy9lOZBmtZFcJuSlhzle5nOABRT4sNCUFNx2Ghbauy1YvYcEraPzVuQNG5F2lo7cvs6Oq8U3VZuTlBYSrrY41fHoszRsXtqWFDJ3m6WxDzLF%2B0iyNHbTYRB77xXQu4EmBo3vNUvDrYkgU5huPz0lhiOvy9IQE3XsfOWQmRqbVlnSZIEk%2Fuk0WeCIYwJcIPpOAk%2BTUznJBOtuA0%2FHPayP2rilX9V93fgop%2BmUCyqfR%2BKjHJGcvdJQar2h3MZQ2gc3lIellCSg0xyWV1o1OisN8PZNNpsqMqUYx6LI9oGtr6JYg62ihKNC0zmu0ME5yJBHBU3D4NB8XfZl33A2TbgcmZelhl2WtV2gZxKlaZ6JrXXBmLGSBcKKLz4XH7Sin08%2Bdn98iHXH7XrarMlW72vND7tiwHL%2BtbiRln9Zuwxf%2BuWlmEQmgyLNYx5uDUY627GfbPgMZBq%2FDIyE%2F0rXG24tR%2BSKRVR3gBjOU%2F%2FQtMZ%2FJlxiIzudLeD2iyXW5F%2F5GX8Nl24zdKMtyCRrJX7dODOFltyftJ4p7CJmpir5onunU%2BZdccTi55Q8cf7lhHme5yEw1pLSnLxQ4tKzUB54rd4XSulLeyT00POxDCZ5i7dd4zXxZpuj%2FW7TYnH3uHIJrjgWdhNEQZZ%2FwZHoKSnVqdXoTUzSq3VD3mRKPi7YrVqv78YsNh6Oz8oParyZT3iOUZxRY6S2hZ%2FFTWfSqUoeLKpUFTE5MFxiVQ0hVVxlQ6tOPGxRXTa1Y8mqdvb02WO8mSCUVQulcGwxLUaY9eH%2FAQ%3D%3D)

[RAG Evaluation Metrics](https://viewer.diagrams.net/?tags=%7B%7D&highlight=0000ff&edit=_blank&layers=1&nav=1&title=AI%20Concepts.drawio#R5ZjBcpswEIafxsd0AGEbji52kkM607EPaXvTwAJqBcsIYZs8fYURBkztTDJO7SQnsx9aCe3%2FC8mMiJds7wTN4m8YAB9ZRrAdkfnIskyDTNRPRcqa3NiuXZNIsKBmRgtW7AmaVE0LFkCuWY0kIpcs60Mf0xR82WNUCNz0m4XIgx7IaAQDsPIpH9JHFsi4ps7YaPk9sChuRjYNfSehTWMN8pgGuOkgshgRTyDK%2BirZesCr6jV1mZnBfTKf%2BfndY%2Fyw%2FLV%2BSovbm7qz25ek7KcgIJXn7ZpYem6ybAoGgaqfDlHIGCNMKV%2B09KvAIg2g6tZQUdvmATFT0FTwN0hZajPQQqJCsUy4vluPWQ10oNEzE9TtciyED6dmpU0rqYjgVIdkutdRrQDABKQoVaIATiVb9x%2BPaidG%2B3Y6dSYELTsNMmSpzDs9f6%2BAaqBXlW3o59NryiRTsy%2FdQYLpnE5QF%2FUzNFFnMi3a%2BeEl3iCf2xvOu%2FAGsS7iDftze8O9Em9YJ71hT04nvJE3xpfwhnKAKH9U%2BV%2FGTfhTd7cL5tteVOroajxlG%2B%2FCU5NnEt7IU3rQNeWFrsRydrdSZFEhVSBMqzObGob5%2BcB%2Frbsqq2xiJmGV0Z1uG3Xu7TspZJx7yFHsckkY%2Br7rKp5LgX%2Bgc4dMiEuCvYfWICRsX%2BGioebNu92c9mttuXptbdqjrDl1ahZ3jrET47hPOkq9QojpQIhZmm9AKLYEDmua%2BuWZ6x%2BGjnOR%2BpuHe6tt%2FKP%2Bzf%2BDbv0bv56%2F%2Fs7x%2Bq9YwjgVTH4YAazDDewKBHAHAniYqnnLj7gCBtvD5QWwjeMrwEMhwJcp5OfeAy6mwNj%2BjwqosP2eUe%2Fd7WchsvgL)

[Model Selection steps](https://viewer.diagrams.net/?tags=%7B%7D&highlight=0000ff&edit=_blank&layers=1&nav=1&title=AI%20Concepts.drawio#R7Vtbc9o4FP41zOw%2BhLElXx%2BB0KRdmGZDdtPum8ACuzEWK4sk9NdXsmXAHJNACZAh8JBYx7rY5zs3fYIabo2frziZhF0W0LiGjOC5hi9rCJkGduQ%2FJZnlEsfQghGPglxkLAS96CctRmrpNApoqmW5SDAWi2hSFg5YktCBKMkI5%2Byp3G3I4qAkmJARBYLegMRQeh8FIsylnm0s5Nc0GoXFyqah74xJ0VkL0pAE7GlJhNs13OKMifxq%2FNyisVJeoZfulzv7m%2FHYbn%2F52m7%2BsL5%2F%2Fvc%2F4yKf7NM2Q%2BavwGki3njq4t3ErFAYDaT%2BdJNxEbIRS0jcXkibnE2TgKppDdla9OkwNpFCUwp%2FUCFm2hjIVDApCsU41neHURy3WMx4tiIeBLTv9aU8FZw90KU72ME%2Blgs186dUj7aC6isq0f1SNuUD%2BlI%2Fcw6o9ATKxlTwmRzIaUxE9FhelWiTHM37zYfesEg%2BDzK0%2ByDPqjs4H6X9x%2FPKkwjCR1TocQv45MXSgyxEGajbAKyf%2BZHEU%2F0WzWkkfUiK0rr825zOgAksAFZoPYWRoL0JyRT4JKPE24D5SLmgz78BJ0RJz%2BLjVWUjHaueFp5vFu4cLnm9Y6xHtoTI1upHx%2FAvqUQ%2B%2B6bG1%2B2i%2BV1PlzUun0utmW69Q7%2FUeOZe8lJHe0cH3g1mDLysra7l0iyR8j5JqXK5rBFTElCehtFE3WGEB%2BmpeCAy3VUX9I1ju6B1TnFv4iHVKc7GDoi65oFznL2F9wmSPpyKu2ELuJtpH93fHIBGlyWRYDxKRqeiecuFtYbpH1vz7sep9VynpHyM3GMr3wPKB7qWO8iJuhxMeTxrcjJ4UDXNa0ovIxSTPo1vWBplwQ1fxnSo5lAajuSut7NyexwFQZbRSByNlIDnupgPaGj5vOPeagPDr9tl0IrUvARaFWYe2hdmPsDs64Qmi2yKnFjpKoge5eVIXer7l0SQ4q5ceqnD%2BjH3%2BSttOqqgYjrRgCapfBbjj%2B7nuxqSr2o0pLGESoTqRi65uulkHt6TL9Gq1%2Bt%2FrltnxSQlyKJsboWhrBgWsBNO0%2Bgn6WcTqSJpotJ7hpDdrNmXaiZZFaV5gWSCIJGwhK5EFC3amwEiBxggDNnI8aAFuvuKGoW5l5Jlhvsbhumh%2FHjeO9mSO96xIzWClEiPcvnu6gFYKrIyxRCZf41PCQwHrYLhWhVg%2BAcF4ygESW3Pey2Eq4HYca%2BFLRvQicgsgtqBdlsIch2Nm8%2FKY5hSjfKk03QfjOoFeVso34Dug6x5mbPsQMjflwNZm9edw5g%2BN9SZSsb8BfrychCTNI0G66pOY1eHgRpd0pddEW0K2Y7u4puAjFplJnJPBr4CpwJ5zCsOxg7ldpDkOOOsB3iABgHobAq06YEQ6xsHDrCQQLml%2F09pKgDg66t3uWcQsqI5pfrd9qA7w%2Fq96FIKvs56k9ot%2BELG5ZamE5bt1z4yVqYFWGHfgIdxLjokWJChuQ%2BZLvrTrGaJ1D9VzeBPHxs%2FbIG9gmd4RWFT2rpVFTt7gxASNmuT4Ick2RzXLYFWyXBUZOL5DvzNIcOQ4Wjcti5aVYxXt3Pxz8b82LVcnPSeyGjjEXd8KsLhNP67sfGQ%2ByhhV5xI09h4yFWve%2BH9tTP9doJBxfFR2TgrSHuvIpz43r5sEzJB1zLwG53shD47mv%2FYiNnuCmJFTljOAMYhMwBGALKeIMmQKajOwM2Bs80ScLZdQfThgwIHaaNz6i5FR6%2Fsa7YJT%2FIPnLohtdRt3F1X5LwOHZG4SZNBuNWhWJOxhy2ScV1%2Bznn19bxqY1inF10OcqqFIVd127gCIG3HA7NEaD7ftCEvPBhQezg8Ci9smuBLvpZ77EMuDEmkF%2FRvbKD%2Fd8PDKx4I%2BUuf8qGujY%2Bue8gKnYruTReQcO9A35DYuaN8fNHMv%2FfnkLHSZ9JPJ9V55VY%2BRUTl6HV9z8lmuUwBBmA5FcSQ50ALsPb1%2FWsMeaH2uE%2BDIEpGZzPYTyAA5yTVdlAZCn7DEGRz8VOr%2FKhl8YM13P4F)

[Chaining](https://viewer.diagrams.net/?tags=%7B%7D&highlight=0000ff&edit=_blank&layers=1&nav=1&title=AI%20Concepts.drawio#R7V1fc5s4EP80eWwG9A94TJzmrp12mpn0rnf30sEgbC7Y8mHcOPfpTwIpNgjHNFjI5xinY2uRQOxqd3%2B7kugFHM3Wv%2BThYvqZxTS7AE68voA3FwC4DiT8S1CeKgpxJGGSp3FFcjaE%2B%2FRfqlpK6iqN6VLSKlLBWFakizoxYvM5jYoaLcxz9livlrAsrhEW4YRqhPsozHTqtzQuphXVx86G%2FitNJ1N1Z9eRZ2ahqiwJy2kYs8ctEnx%2FAUc5Y0X1a7Ye0UwwT%2FHltxB9%2FBrcf7%2FL%2F0LF%2FCMef%2Fn%2B%2B7vqYrc%2F0%2BT5EXI6Lw57aSAfrXhS%2FKIxZ58ssryYsgmbh9n7DfU6Z6t5TMVVHV7a1PnE2IITXU78mxbFkxwL4apgnDQtZpk8y58if%2FpDtL%2FEqvinvFxZuFnXSk%2ByVPVVdLAh2j18kfWWbJVH9IV6WA7PMJ%2FQl67nPQufaw1lM8o7ydvlNAuL9Ee9c6EcvpPnehsR8R9SSj8hMWhDYoNwvgdHZdM7lvLuAEcaL%2Bijqok0XS4B9UtUkpatGnJ57sbrRYVOVlQdlMR1bWqJ7OWPMFvJO43YbJHRNSd%2BDZcPmmQ2fBdMfJymBb1fhCUnHrljrPM4YfNCCkA843WSZtmIZSwvrwWTJAFRxOmTPIxTzvbauTgghJ9bFjl7oFtnYjImmDxL8AfNC7p%2BhQx1lqurEFJXB78qPm7cIlLefrrlEoljSEjk7HwaTqWLXkGbeuVpelVpk%2BNqsqxLao9GtWjK7e1oFAQG9QE4XsM9QE0hXAX%2BBlEI%2FxQdRmDIt9eFB51gUN8e7FIEHVEfvSJAcGyKoIJHa64BHJVvcFXgvM85AGLTOahu6kqhBy1HrxQIHZ1SWAnWjY%2Ftvnim3T8A0vAPBA%2FqH1yo6cIFIFkhkP8inNfESP5ZiRzSNR%2FBxbtq9F%2FxCnMmNWFTg%2F%2BaiO%2B7nEczRZlKE6k5VtoGhcOqu%2FBeVzeqWryggF0CnnqAE4fUT3YEOB4NCXXaAhwS%2BXScDAjoACbdVBb6plT2JFMAbuccgNVMmasnAT59%2BqwJZPlAi2gqub0Q9qDsCL7mf3ysjKp%2FmFcdCYqABjqxjebpRFevxr%2Fctjs0iW00Tye6ejVRUr2uE9toHtZ73GzttrR2G635Hx%2B8qyJL59wIqNy%2BI7MnW2aBf26FVJtGxnGvfP%2BmzZAk5cHPxOFy%2Bqwowp6kEVeTcEyzO7ZMi5TN%2BbkxKwo226pwlaUTcaIQ6nQdylLEdYPmO%2FM8LlBlOXLELcPlonqsJF2Lflwvp%2BFCnJytJ2I25TJ8XKLLnFYa9SES%2FbnmxepXvdaYxjmLHkyaRxhc4oZT1A0kxrp9xNiUilpJARm3j11zOaBvjrRsepXn4dNWBWnDdofOANRGAZZe6rZjfeiRhuCrHhwWP%2BmZpi%2BrYrEqWnNN%2FILpYkn3AxljuoWcGo9QgDtpFvRMaZav8a8H25qmuTx0U35bHibZ7NVndADQc9gA6mwGyBSb9TzQIDAfvGGYD6Fbt0dux8jcGMwHVtJVpt3YYbyT7k4CWDeVSmEHisxVj7dU9sO8dCxvQXXqvIfYs606J5nUUkN6PwLsu%2FKhH%2FP1JNU5Qj5HyG82QoYEX4J6iIwUdLQVIgMrKcTNVFhtHmyzZMLSTJiSzn7D6ls1rHrq8Tl6bXF4tqNXMe69%2BrjHWAfVgwawgGgsHCSygm84skJeY9IM6cZvYHjonSQ87Guc2iMrDIO678KNtIfpyErPOb2ZyAp5dd5DX89MDaw6wSmqDnQ6AgBodf2x6uY5sjpHVufISswxuVpkpeCkrchKpZHbQHrLLhvbIF2wsAnS%2FW7BqTGQDoHGwh7hzbHMMiEVo6lZJuXbbc0ywf%2Fjpq%2B9Hho57VLovTI8aJoaz21MMVRG0RgQhkhTCxVwHhAJJ1h82pHwuPy0IuHyMJlMAKTJf6LgkLW1sxDb0KDX7jVqhjyY%2BjFqF3TgxQ6P2lsE7YMx3BK0QUhOuiq81a1LUE8olZC8ZTnJGZafYfmbgOUA6b4S6bYa%2BbqpRqbyFtBKyu9ApjrxI7pr6y8Nybh96%2B%2FYxwgPMX2ikk37TTWyaqr1RGJlqvXg4myqhzbVOwb87kF9mib5JQU3Bq0JOD5zbSXNfKy7%2BJU09htYqwt%2FVDc1A6snGs4G1rKBTaIY%2BG0GNvaCsfOmDGyShNhxDBpYpC8A8pC%2BEGJQA4ta1u0fr4EtS3c0T%2FnDi3E0iNVV%2Bbb9k4J998e%2FbsONr2XEPLmZe9eeG%2BDBPU3MbLtBLen782B75WCzugQN6Wt7D7GBykRq0kRew2uqjw%2F1pe4YtNhxYwLRJwUOsSbQRALCROTiHJ9AXlik2WP%2Bdxd%2BewnP7MZ25pAOvGzKQ5%2FlHFYeJ7kZGPV9B8IOTBFoAMF13MZlTL%2FfUt%2Bje0OjdCkCjqYke0108sNvDYEg4XyIh3UkgXoPz%2FZqgKBFUdSc4%2BEVxcp74rZAmkv8Gky7dHxyjFANdd2G1VtJ%2B8nTdgoNeeTi2GE36gq77cpSvVfcmiy9EnJv6yb2jlKeXXfyWJanvkjwNuWi4aSr%2BfKRs%2BvVQNEY1oZaEotDA32xZesKHPTzSwV5cfMS%2FApbbP4rAfj%2BPw%3D%3D)

[AI lifecycle project](https://viewer.diagrams.net/?tags=%7B%7D&highlight=0000ff&edit=_blank&layers=1&nav=1&title=AI%20Concepts.drawio#R7VrZbts6EP0aA%2B1DA1GytsfYsdMWLhrAtyhw32iJlthIokHRW7%2F%2BkiK1UnYTxEuK6wRBpMPF5Jw5MxwmA2uc7h4pXMXfSIiSgWmEu4H1MDBNYFgO%2FyWQvUQ%2BDf2hRCKKQ4kZNTDHv1E5VKFrHKJcYRJihCQMr9pgQLIMBayFQUrJtt1tSZKwBaxghDRgHsBER3%2FikMUS9Wyjxj8jHMXlJwNDtaSw7KyAPIYh2TYgazKwxpQQJp%2FS3Rglwnptu0wPtFYLoyhjLxkw%2Bf4duL%2B8rzP6ZeM%2FTv8N7%2FfBJ9tXi2P7csco5AZQr4SymEQkg8mkRkeUrLMQiWkN%2Flb3mRGy4iDg4C%2FE2F6xCdeMcChmaaJa9aWr3eRkTQN0bL2e7CgW2RipdvyISIoY3fMOFCWQ4U2bRaicIar6VUOfCOZrMQ3luaZhySHKb4Hrme05GKQRYmpYbXb%2B0FhHDRVkvIYYtdENTNZqEz8yRiHOuOFNY%2FZN461mRZh4G2OG5itYWHPLxflCBjaIMrQ7atqy1elYyFYW2tY6AaXzxw2NOMZhNlpmfK3N3OHf5cylIaQrHduYdWqvf5Ody3U3fHM%2B%2FefqDmlZ9rvzSOcv88iTO1p%2FeHXLvHOt8FputOHCTxQtETcZNxCX5DrDWTQwHZgKR80W%2Baoy7BVd3Cm995iL%2B%2BCCHu7Y1%2FDwEOZxZfY3uPvwQu7eSZWe2aHj3N4%2B1LxdY61hXeGPmB%2BBZ3CBkieSY4ZJxtsWhDGS8g4wwZEAAm5xRDmQiJ4jGDxHBbVjkhBaTGsti6%2FGpPdqLBNUj3JGyXN1sjYrpDGDYXjGVPhJdYA2mh6gWlZiG%2BkuEmXIHSa5e4d5SZDfhSRYp3yZ%2BWkEOPTaTPqGrj%2FbuaT%2BHI3aH1lBR3FODCGDGtV8r6ytJ43QLlspDkMpXZTj33BRTCVMvxI%2BW2zKHg3sBzEXV2suhQs0PjOSiVmWOEk60AnIMbvkOJ4eHH2dHNM5FzmuRs6McGuZBllW5DiJICPEG%2F4YiccPc5SI5vl6hegG5yj8qCciOYqvqjHwf0u0Z7n6QQ9YOtPDsx2Jr1JHnyoNXqiotnzzynlQL6o12vJnxIK45GTNEl5xj6vLJQEuScYa%2BYl%2FT8UaRjz3hBjVbT0S4N0nLrDd0SG9tNLaH%2FNwf07VNN7wGLF25U7ALN%2FV5sVHwnwlN7rEO7EOKXtEJxuZQ6XSO%2BkWbvPhqROt5XROuj7QM215rmlK3PXOJHG3p%2BpNYZJwE3MHEjY9EtGrUC7ubhCkorq4RfUO5d3ixuur362eBH62sO7q8eFd1%2B8uuFAktzqR%2FNL1e7nRhhh57BIS%2FDAV5ociVH7skWJXc%2BpPFi8SXis1tJUz4q47lj9cReZYIHem3QP2Ya4OAr0b%2FwX6PqEL9mGuDgK9m3grV90G%2BzDX1lfcHQ16RoPO6CLsvCrTdlKqAe4976Gvdqyqz3eXV3tzKI%2FOhcK%2FBGI9IljLp3avBQopCZ5PdJQ2tEtTvWgqq95WzD1bnr3KjdIbYu6F7kwd4F455up3plMuWXlbWpxsyr%2F%2FAo3BW%2Bi8dOhsh8jlMgh8vy9EWo7lW%2BEtRB4OkbZ%2B6X7tEKnfKx1Q4h%2Fq25sSL67EIEB2cSS5KfHVSnSdbgo8nxL5a%2F2%2FOTKF1v%2FiZE3%2BAw%3D%3D)

