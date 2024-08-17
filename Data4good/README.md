### BAN 5753-fall2023 - INCISIVE

# The Purdue Data 4 Good Case Competition 2023

### ***Extracting Medical Insights: A Comprehensive Approach Using LLM-Based Logic***



> ## MEET THE TEAM
+ **Prafulla Balasaheb Sature**
+ **Abdul Mujeeb**
+ **Anirudh Bommina**
+ **Tharun Ponnaganti**


![team](https://github.com/osu-msba/ban5753-fall2023-incisive/assets/144861021/c1cb6abe-a389-4ba5-b85c-ada9b7788530)


### Executive Summary:

+ In tackling the 2023 Purdue “Data for Good” challenge, our focus revolves around addressing a pertinent data extraction problem inherent in anonymized or synthesized transcriptions of medical conversations and dictations in multiple languages like Spanish, Urdu, Hindi etc., The competition mandates that teams, including ours, leverage logic and chains based on Large Language Models (LLMs) for tasks such as information extraction, rephrasing, summarization, and validation. Specifically, we aim to utilize open access and permissively licensed LLMs, in our case,  Nous-Hermes-Llama2-13B  to generate predictions tailored to this competition's requirements. Our approach aligns with the competition's objective of providing solutions applicable in private, regulated healthcare environments. Our code framework embodies a methodical approach to the processing of medical transcripts. It was successful in extracting critical information such as patient names, ages, medical conditions, symptoms, precautions, and medications with an accuracy score of 1.98. 
+ Our participation in the 2023 Purdue "Data for Good" competition is centered on a two-fold approach, wherein we dedicate half of our resources towards gaining expertise in Microsoft's Azure technologies while simultaneously incorporating sophisticated Large Language Models (LLMs) into the mechanization of medical documentation procedures.

### Introduction:
+ In the ever-changing field of healthcare, our team finds itself at the intersection of technological advancement and data-centered transformation, poised to unravel the intricacies embedded in medical documentation. The Purdue University-led collaborative initiative known as the 2023 Purdue "Data for Good" competition, in partnership with Microsoft, INFORMS, and Prediction Guard, serves as the backdrop for our pursuit of harnessing the potential of data for impactful contributions to healthcare.
+ At the core of this competition lies a profound challenge—to streamline and optimize the often-laborious processes involved in medical documentation. The extraction of vital information from medical transcripts becomes the focal point, a task that has traditionally burdened healthcare professionals. Within this context, our overarching objective is to deploy technology through a focused strategy on  LLMs, automating the extraction of vital data points such as patient names, ages, medical conditions, symptoms, precautions, and medications..
+ The problem statement is evident—medical documentation is a labor-intensive endeavor that demands meticulous attention from healthcare professionals. The technical foundation of our solution, encapsulated in a systematic code framework, aims to ease the burden on healthcare professionals by automating the retrieval of essential information points, ensuring accuracy, and upholding privacy standards.

### Objectives:
+ Application of LLMs in Healthcare Documentation:
  +Preprocessing with LLMs: By employing Google Translator, we engage in tasks such as translating non-English transcripts into English, thus establishing 
   standardized language for subsequent analysis.
  +Information Extraction: Through the utilization of LLMs, particularly Llama 2, we are committed to extracting vital medical details, including patient 
   information, symptoms, diagnoses, and prescriptions, from synthesized medical conversations and dictation
+ Mastery of Azure Technology:
  +Azure Fundamentals: Our aim is to enhance our comprehension of the fundamental aspects of Azure's cloud computing, ensuring a strong grasp of its capabilities. 
  +Azure AI: We aspire to achieve proficiency in harnessing AI services and functionalities on the Azure platform, navigating the intricate realm of artificial 
   intelligence.


### Data Sourcing:

![Transcript_snippet](https://github.com/osu-msba/ban5753-fall2023-incisive/assets/144861021/f3ef8322-1ef8-4db1-a7dd-312d07902a3d)

+ The core of our project relies on the curated dataset provided by the 2023 Purdue "Data for Good" competition organizers. The dataset comprises anonymized or synthesized transcriptions of medical conversations and dictations in different languages—typical scenarios encountered in healthcare documentation. This diverse dataset serves as the foundation for our exploration of extracting valuable information to fill out medical forms.

### Data Visualization:
+ Word Clouds aimed at providing more information, clarifying unclear questions, and making corrections in case of errors. Here symptom, condition, medication are the most occurring words. It also has some words in multiple languages which should be translated.
  
 ![wordcloud](https://github.com/osu-msba/ban5753-fall2023-incisive/assets/144861021/39f328be-e531-4289-9797-1d5f263e2237)
 
   
### Data Preprocessing:
+ The traditional approach of using tokens and eliminating the stop words is not necessary as the LLM model 'Nous-Hermes-Llama2-13B' can efficiently handle that. But the dataset presents a unique challenge with transcripts in different languages like Spanish, Urdu, Hindi, etc., also different styles, and medical jargon. To address this, our preprocessing pipeline incorporates a translation step for texts not in English. Leveraging Google Translate within our Python environment, we dynamically translate non-English transcripts into English. This not only homogenizes the data for uniform processing but also ensures that language models comprehend and extract information accurately.
  
### Modeling Approach:
+ Information Extraction using LLMs:
Our decision to employ 'Nous-Hermes-Llama2-13B' was underpinned by its expansive parameter count, making it well-suited for the nuanced task of medical information extraction. The competition's emphasis on diverse medical transcripts, spanning different languages, styles, and medical jargon, necessitated a robust model capable of comprehending and extracting information from complex contexts.
+ Handling Exceptions and Ensuring Robustness:
Our code incorporates a robust exception handling mechanism, addressing potential errors during transcript processing. In case of an exception, it prints an error message specifying the problematic transcript. A placeholder entry with 'None' is added to results, allowing the script to continue processing other transcripts.
+ Subset Selection for Testing:
Initially, a smaller subset of the first 100 transcripts is selected for testing, facilitating rapid debugging. Recognizing the need for a more comprehensive evaluation, the subset is expanded to include the first 2001 transcripts, ensuring thorough testing across a diverse range.
+ LLM Prompt:
Our modeling strategy places a significant emphasis on the construction of a detailed and nuanced prompt for Large Language Models (LLMs), particularly centering 
around 'Nous-Hermes-Llama2-13B'. The prompt serves as a crucial guide for the language models, instructing them on specific tasks related to the extraction of 
medical information from transcripts. Crafted with meticulous attention, our prompt outlines a series of targeted instructions, including the identification of 
the patient's name, extraction of age, discerning medical conditions, listing symptoms, capturing precautions, and identifying medications or drugs mentioned in 
the text. This tailored prompt not only enhances the LLMs' understanding of the tasks at hand but also ensures a focused and granular extraction process. The 
inclusion of the {transcripts} placeholder within the prompt facilitates dynamic interaction with varying medical dialogue structures, enabling adaptability to 
diverse scenarios presented in the competition's dataset. This detailed and task-specific prompt design is integral to our approach, enhancing the precision and 
effectiveness of information extraction from complex medical transcripts.
 
 ![Prompt](https://github.com/osu-msba/ban5753-fall2023-incisive/assets/144861021/7fb1700e-882d-40b4-8067-68905fa0bfc0)
 
+ Results Storage and CSV Export:
An empty list, all_results, is initialized to store the extracted information. The code processes each transcript using the process_batch function, and results are appended to all_results. Extracted information is systematically saved to a CSV file with columns for unique identifiers (Id) and extracted text (Text). This organized storage facilitates further analysis and evaluation of the extracted medical information.
Our modeling approach is not only centered on the capabilities of LLMs but is fortified by robust exception handling, thoughtful testing strategies, and systematic result storage, ensuring a holistic and effective solution to the competition's data extraction challenge. 

### Results and Performance Evaluation:
+ Our choice of model 'Nous-Hermes-Llama2-13B' has yielded impactful results. The application of Large Language Models (LLMs), particularly centered around the adept Llama 2, showcased an exceptional ability to extract vital medical details from a spectrum of transcripts. Here are the salient features encapsulating our journey and outcomes:
  
  ![result_snippet](https://github.com/osu-msba/ban5753-fall2023-incisive/assets/144861021/7c54f094-c0c5-49e9-976a-de82a77ffb0e) 
   
+ Extraction Precision:
The LLMs, meticulously guided by our precisely crafted prompts, exhibited a commendable accuracy score of 1.98 in extracting intricate medical details. This includes patient names, ages, medical conditions, symptoms, precautions, and medication specifics. The tailored instructions embedded within our prompts played a pivotal role, fostering a nuanced understanding of the complexities inherent in medical conversations.
+ Ranking and Recognition:
Our solution earned us a noteworthy position, securing the 46th rank amidst a competitive landscape. This standing not only underscores the efficacy of our approach but also positions our solution as a competitive force within the Data for Good domain.
+ Azure Certifications for all the Team Members:
  All the team members successfully completed the Microsoft Azure Fundamentals Certification.

  ![Azure_Certification](https://github.com/osu-msba/ban5753-fall2023-incisive/assets/144861021/65d9515f-9bc2-4c59-8fc9-05b43d547336) 

### Generalization/Explanation:
+ In the "Data for Good" competition at Purdue University in 2023, our team utilized sophisticated language models, with a particular emphasis on Nous-Hermes-Llama2-13B, to enhance the efficiency of extracting information from medical transcripts. Through the meticulous construction of precise prompts, we guided Nous-Hermes-Llama2-13B to achieve remarkable precision in retrieving vital details such as patient names, ages, medical conditions, symptoms, precautions, and medication specifics. Our approach, bolstered by robust handling mechanisms and extensive testing, demonstrated the adaptability and scalability of our solution. This successful integration of advanced language models positions our solution competitively within the Data for Good domain, providing a glimpse into a future where healthcare processes are optimized, and technology contributes to a positive societal impact.

### Future Scope:
+ Refinement through Continuous Learning:
Continuous refinement of our model through an iterative learning process is a key aspect of the future scope. Engaging in ongoing model training and reevaluation with additional datasets can enhance the accuracy and adaptability of the system to evolving patterns in medical transcripts.
+ Integration of Multimodal Data:
The inclusion of multimodal data, such as audio and images alongside text transcripts, represents an exciting avenue for future development. Integrating diverse data types can further enrich the information extraction process, providing a more comprehensive understanding of medical interactions.
+ Privacy-Preserving Techniques:
Implementing privacy-preserving techniques in the information extraction process is crucial, especially in the healthcare domain. Future efforts can explore advanced cryptographic methods or federated learning approaches to ensure the confidentiality of sensitive medical information.
+ Real-Time Application in Healthcare Settings:
The development of a real-time application for medical information extraction holds immense promise. Integrating the solution into healthcare settings could significantly reduce administrative burdens, allowing healthcare professionals to access structured data promptly during patient interactions.
+ Extension to Other Domains:
The robustness of our information extraction approach can be extended to other domains beyond healthcare. Exploring applications in legal, educational, or business settings where structured data extraction from unstructured text is essential could broaden the impact of our solution.
+ In essence, the future scope of our work extends beyond the confines of the competition, delving into continuous improvement, diversification of data sources, and practical deployment scenarios. 

### Conclusion:
+ In the domain of Large Language Models (LLMs), our primary focal point has been harnessing the potent capabilities of models such as Llama 2. These sophisticated LLMs function as the foundation of our methodology for automating the extraction of crucial medical information from a diverse array of transcripts. Our focus on LLMs coincides with the cutting-edge advancements in natural language processing, empowering us to tackle the complexities inherent in medical discussions and transcriptions.
+ The strategic implementation of LLMs, particularly Nous-Hermes-Llama2-13B, has played a pivotal role in our competitive positioning, wherein we have attained the 46th rank. This accomplishment underscores the efficacy of our approach in processing and interpreting medical transcripts to derive valuable information.
+ Our meticulous prompting strategy involves crafting well-structured and contextually pertinent prompts that guide the LLM in generating precise and meaningful responses. This strategy is complemented by exceptional handling mechanisms that ensure the robustness and dependability of our information extraction process. Moreover, extensive testing on varying scales of datasets has been a cornerstone of our LLM-centric methodology, affirming the adaptability and scalability of our solution.
+ As we systematically accumulate and analyze results, the significance of LLMs in our overarching strategy becomes evident. These models embody a technological frontier in comprehending natural language, empowering us to navigate the complexities of medical terminology with accuracy. 

### References:
+ Bisercic, Aleksa & Nikolic, Mladen & Schaar, Mihaela & Delibašić, Boris & Lio, Pietro & Petrovic, Andrija. (2023). Interpretable Medical Diagnostics with 
  Structured Data Extraction by Large Language Models.
+ Singhal, K., Azizi, S., Tu, T. et al. Large language models encode clinical knowledge. Nature 620, 172–180 (2023).
+ Choi HS, Song JY, Shin KH, Chang JH, Jang BS. Developing prompts from large language model for extracting clinical information from pathology and ultrasound 
  reports in breast cancer. Radiat Oncol J. 2023 Sep; 41(3):209-216. 

