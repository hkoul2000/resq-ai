# Literature Review: Uncertainty-Aware Multimodal AI Framework for Urban Disaster Prediction

## 1. Introduction
The increasing frequency and severity of urban natural disasters, particularly floods, demand rapid, reliable, and actionable intelligence for emergency response. Recent advancements in deep learning (DL), remote sensing, and crisis informatics have significantly improved disaster mapping and prediction capabilities. However, translating predictive models into actionable emergency logistics remains a critical challenge, primarily due to the lack of principled uncertainty quantification (UQ) and the disjointed nature of prediction and decision-making pipelines. 

This literature review systematically examines the intersection of deep learning for remote sensing, multimodal data fusion, crisis informatics, uncertainty quantification, and emergency resource allocation. The scope encompasses over 60 recent scholarly works (primarily from 2019-2026), structured into six core thematic areas. The review evaluates current methodologies, highlights their limitations, and identifies the existing gaps in end-to-end, uncertainty-aware decision-support frameworks. The findings motivate the proposed research questions and contributions for the ResQ-AI framework.

## 2. Flood Mapping from SAR and Optical Imagery
The foundation of modern disaster prediction relies heavily on Earth Observation data. Synthetic Aperture Radar (SAR) and optical imagery represent the primary modalities for flood inundation mapping. Deep learning, notably Convolutional Neural Networks (CNNs) originally inspired by the U-Net architecture (Ronneberger et al., 2015), has become the standard for segmenting flooded areas.

The release of large-scale annotated datasets has driven significant progress. Bonafilia et al. (2020) introduced Sen1Floods11, providing global georeferenced data for training algorithms on Sentinel-1 SAR imagery. Similarly, Rahnemoonfar et al. (2021) developed FloodNet, focusing on high-resolution aerial imagery for post-flood scene understanding. These datasets enabled the development of highly accurate models. Nemni et al. (2020) demonstrated the efficacy of fully convolutional neural networks for rapid flood segmentation using SAR data, while Konapala et al. (2021) explored the diversity of combined Sentinel-1 and Sentinel-2 data, showcasing performance improvements. 

Advanced architectural designs have further refined flood mapping. Zhao et al. (2021) and Bai et al. (2021) integrated spatial-temporal attention mechanisms and multiscale attention into Fully Convolutional Networks (FCNs) to better extract complex flooded areas from SAR series. Peng et al. (2019) introduced residual patch similarity learning for urban flood mapping, addressing the high spatial heterogeneity of urban environments. Furthermore, deep learning techniques have been widely reviewed for flood mapping (Bentivoglio et al., 2022) and specifically for SAR-based detection (Katiyar et al., 2021), confirming the paradigm shift from thresholding to learned feature extraction.

Real-time and large-scale applications are the ultimate goals of these systems. Shen et al. (2019) and Zha et al. (2023) focused on near-real-time inundation mapping frameworks, crucial for immediate response. Global environmental monitoring models (Mateo-Garcia et al., 2021) and high-resolution spatial models (Huang et al., 2022) illustrate the scalability of these approaches. Moreover, spatiotemporal analyses (Munawar et al., 2021) and semi-supervised active learning methods (Li et al., 2019) address data scarcity and temporal dynamics in urban settings.

**Section Gap:** While state-of-the-art models achieve high accuracy on standard benchmarks, they predominantly assume deterministic inputs and outputs. They struggle with the inherent noise in SAR (speckle) and optical (cloud cover) imagery and rarely provide confidence bounds on their flood maps, which is dangerous for downstream emergency response.

## 3. Multimodal Remote Sensing Fusion
No single modality can provide a complete picture during a crisis. Optical imagery offers rich spectral information but is obstructed by clouds and darkness; SAR penetrates clouds and operates day/night but suffers from speckle noise and complex layover effects in urban areas. Multimodal fusion aims to leverage the complementary strengths of diverse data sources (Hong et al., 2021).

The curation of multimodal datasets like SEN12MS (Schmitt et al., 2019) has spurred research into cross-modal learning. Ienco et al. (2019) and Interdonato et al. (2019) utilized multitemporal Sentinel-1 and Sentinel-2 data to improve land cover and change classification through deep recurrent and convolutional networks. To handle missing or corrupted modalities, techniques like cross-modal knowledge distillation (Wang et al., 2022) and deep image translation (Zhang et al., 2020) have been developed, allowing models to infer features of one modality from another.

Fusion strategies often involve complex architectures. Hughes et al. (2020) designed networks to identify corresponding patches across SAR and optical data, facilitating robust alignment. Wu et al. (2021) proposed cross-modal learning specifically for remote sensing classification, and Saha et al. (2021) applied unsupervised multiple-change detection across modalities. Paoletti et al. (2019) extended deep learning fusion to hyperspectral data. Furthermore, general conditioning layers, such as FiLM (Perez et al., 2018), offer flexible mechanisms to modulate features from one modality based on another, enhancing the representation power of fused networks.

**Section Gap:** Multimodal fusion in remote sensing often assumes that all modalities are available during inference. They lack robustness to "missing modality" scenarios common in operational settings (e.g., waiting for a satellite pass). Moreover, these fusion techniques rarely propagate the uncertainty of the individual modalities through the fused representation.

## 4. Crisis Informatics and Social Media
Beyond satellite imagery, social media and crowdsourced data provide critical, ground-level, real-time context during disasters. Crisis informatics leverages Natural Language Processing (NLP) and Computer Vision to extract actionable insights from this user-generated content.

Imran et al. (2020) outlined the extensive role of AI in analyzing social media for disaster response, highlighting its utility for situational awareness. Datasets like HumAID (Alam et al., 2021) have been instrumental in training models to categorize human-centric disaster incidents from text. 

Multimodal analysis of social media combines text and images for richer context. Ofli et al. (2020) and Abavisani et al. (2020) demonstrated that fusing textual features with image embeddings improves the categorization of crisis events and assessment of severity. Nguyen et al. (2020) introduced the LATTICE framework specifically designed for processing disaster-related tweets. Said et al. (2019) even explored the joint detection of natural disasters using both social media and satellite imagery, bridging the gap between ground and aerial views. Furthermore, incident-driven machine learning (Weber et al., 2020) and image-based public utility assessment (Alam et al., 2020) focus on extracting specific operational details, such as infrastructure damage, directly from user uploads.

**Section Gap:** Social media data is inherently noisy, unverified, and geographically biased. Current models often treat this data as ground truth without estimating the epistemic uncertainty or the reliability of the source. Integrating this noisy social stream with satellite data in a mathematically rigorous, uncertainty-aware manner remains largely unsolved.

## 5. Uncertainty Quantification in Deep Learning
In safety-critical applications like disaster response, knowing when a model is uncertain is as important as the prediction itself. Uncertainty is generally categorized into aleatoric (data noise) and epistemic (model ignorance). 

Ovadia et al. (2019) benchmarked various UQ methods under dataset shift, showing that deterministic models often become overconfident on out-of-distribution (OOD) data. Foundational techniques include Deep Ensembles (Lakshminarayanan et al., 2017), which train multiple models from different initializations, and Monte Carlo Dropout (Gal & Ghahramani, 2016), which approximates Bayesian inference by applying dropout at test time. Both are standard baselines for epistemic uncertainty.

More recent approaches aim for single-forward-pass efficiency. Evidential Deep Learning (Sensoy et al., 2018; Amini et al., 2020) parameterizes the outputs as a Dirichlet or Normal-Inverse-Gamma distribution, directly estimating both aleatoric and epistemic uncertainty. Deterministic neural networks with appropriate inductive biases (Mukhoti et al., 2021) and ensemble distribution distillation (Malinin et al., 2019) offer alternatives to sampling-based methods.

UQ is gaining traction in remote sensing and healthcare. Mena et al. (2021) and He et al. (2020) applied epistemic uncertainty estimation to satellite image classification, demonstrating improved reliability. Dusenberry et al. (2020) analyzed model uncertainty for electronic health records, an analogous safety-critical domain. Broad frameworks for UQ (Loquercio et al., 2020) and comprehensive surveys (Abdar et al., 2021; Gawlikowski et al., 2023; Wilson & Izmailov, 2020) consolidate these methodologies. Finally, Conformal Prediction (Angelopoulos & Bates, 2021) has emerged as a powerful post-hoc calibration technique, providing distribution-free, valid coverage guarantees for model predictions.

**Section Gap:** While UQ methods are well-studied in isolation or on standard vision tasks, their application to complex, dense prediction tasks like multimodal flood mapping is limited. Furthermore, these uncertainty estimates are rarely used to explicitly drive downstream operations; they are often merely visualized as heatmaps.

## 6. Emergency Logistics and Resource Allocation
The output of a predictive model must ultimately inform resource allocation (e.g., routing rescue boats, positioning sandbags, dispatching medical supplies). Humanitarian logistics focuses on optimizing these operations under severe constraints.

Given the inherent unpredictability of disasters, stochastic programming and robust optimization are widely used. Vahdani et al. (2019) and Altheeb et al. (2019) developed reliable network designs and stochastic programming models for humanitarian logistics under uncertainty. Tofighi et al. (2020) formulated a two-stage stochastic model specifically for blood supply chains during emergencies. 

Routing optimization is a major focus. Sabouhi et al. (2019) designed resilient routing protocols, while Dufour et al. (2021) applied robust optimization to disaster relief routing. Advanced technologies are also being integrated; Zhou et al. (2020) modeled drone routing for disaster response, and Cao et al. (2021) applied Deep Reinforcement Learning (DRL) to dynamically route relief vehicles. Reviews of humanitarian supply chain management (Mota et al., 2020) and facility location (Boonmee et al., 2021) emphasize the complexity of multi-objective optimization in these scenarios. Chance-constrained programming (Fereiduni et al., 2021) is particularly relevant, allowing decision-makers to set acceptable risk thresholds for constraint violations.

**Section Gap:** The stochastic scenarios used in these optimization models are typically assumed, generated from historical data, or derived from simplistic parametric distributions. They are rarely directly coupled with the complex, non-parametric, high-dimensional predictive distributions output by deep learning models.

## 7. Integration and Decision-Support Systems
Bridging the gap between machine learning predictions (predictive models) and operations research (prescriptive models) is the frontier of decision-support systems. Traditionally, this is a two-stage process: train a model to minimize predictive error (e.g., MSE), then feed the point predictions into an optimizer. 

Recent work advocates for "end-to-end" integration. Elmachtoub & Grigas (2022) introduced "Smart Predict, then Optimize" (SPO), which trains the predictive model to directly minimize the decision regret of the downstream optimization task. Mandi et al. (2020) extended SPO for hard combinatorial optimization problems. Wilder et al. (2019) proposed melding the data-decisions pipeline by differentiating through the optimization problem itself. Similarly, Ferber et al. (2020) introduced MIPaaL (Mixed Integer Program as a Layer), allowing deep networks to backpropagate through integer linear programs. 

In the specific context of disaster management, Cameron et al. (2022) highlighted the urgent need and path towards end-to-end AI systems that seamlessly connect earth observation data directly to emergency response actions, minimizing latency and maximizing impact.

**Section Gap:** While decision-focused learning (SPO, MIPaaL) successfully integrates prediction and optimization, these frameworks are predominantly deterministic. They train models to yield a single point prediction that minimizes downstream cost. They do not propagate full uncertainty distributions into stochastic or chance-constrained optimizers, which is strictly necessary for risk-averse emergency planning.

## 8. Comparison Table

| Reference | Year | Method | Modality | Uncertainty Method | Decision Support (Y/N) |
| :--- | :---: | :--- | :--- | :--- | :---: |
| Bonafilia et al. | 2020 | CNN Segmentation | SAR | None | N |
| Nemni et al. | 2020 | FCN | SAR | None | N |
| Konapala et al. | 2021 | DL Fusion | SAR + Optical | None | N |
| Zhao et al. | 2021 | Attention FCN | Time-series SAR | None | N |
| Huang et al. | 2022 | High-res DL | Aerial/Optical | None | N |
| Zha et al. | 2023 | Real-time DL | SAR | None | N |
| Hong et al. | 2021 | Multimodal DL | Multi-RS | None | N |
| Wang et al. | 2022 | Cross-Modal Distil. | SAR + Optical | None | N |
| Imran et al. | 2020 | AI Analysis | Social Media | None | Y (Informational) |
| Abavisani et al. | 2020 | Multimodal Class. | Social (Text+Image) | None | N |
| Lakshminarayanan et al.| 2017 | Deep Ensembles | Agnostic | Deep Ensembles | N |
| Gal & Ghahramani| 2016 | MC Dropout | Agnostic | MC Dropout | N |
| Sensoy et al. | 2018 | Evidential DL | Agnostic | Dirichlet / Evidential | N |
| Amini et al. | 2020 | Evidential Regress.| Agnostic | Normal-Inverse-Gamma | N |
| Mena et al. | 2021 | Bayesian CNN | Optical RS | Epistemic Sampling | N |
| Angelopoulos & Bates | 2021 | Conformal Prediction| Agnostic | Calibration / Coverage | N |
| Vahdani et al. | 2019 | Stochastic Prog. | Tabular/Parametric| Exogenous Scenarios | Y |
| Cao et al. | 2021 | Deep RL | Tabular | Exogenous | Y |
| Fereiduni et al. | 2021 | Chance-Constrained | Tabular/Parametric| Exogenous Probabilities| Y |
| Elmachtoub & Grigas| 2022 | SPO+ | Agnostic | None (Deterministic) | Y |
| Ferber et al. | 2020 | MIPaaL | Agnostic | None (Deterministic) | Y |
| Cameron et al. | 2022 | End-to-End AI | Multi-RS | Concept / None | Y |

*(Note: Table highlights a representative subset of the reviewed literature to illustrate the methodological divide between prediction, uncertainty quantification, and decision-making.)*

## 9. Research Gap Analysis
The literature reveals a fragmented landscape in disaster AI. 
1. **Prediction models** (Section 2, 3, 4) achieve high accuracy using multimodal data but are largely deterministic and brittle to missing data. 
2. **Uncertainty Quantification models** (Section 5) provide rigorous statistical bounds but are rarely applied to complex multimodal semantic segmentation tasks in remote sensing.
3. **Logistics models** (Section 6) handle uncertainty beautifully via stochastic programming, but assume the probability distributions are provided exogenously and analytically.
4. **End-to-End systems** (Section 7) bridge prediction and optimization but discard uncertainty, focusing on risk-neutral point estimates.

**The Crucial Gap:** There is no existing framework that seamlessly pipelines raw, multimodal data (SAR, optical, social) through a deep neural network that outputs *calibrated predictive distributions* (not just point estimates), and directly couples those empirical distributions into a risk-averse, *chance-constrained emergency resource allocator*. Current systems force a choice between uncertainty-awareness (stopping at a heatmap) and automated decision-making (ignoring the risk of the model being wrong). 

## 10. Research Questions and Contributions

To address the identified gaps, the ResQ-AI project formulates the following research questions:

*   **RQ1:** Does multimodal fusion beat single-modality baselines, especially under missing modalities and distribution shift?
*   **RQ2:** Which UQ method gives the best calibration and error detection for flood prediction?
*   **RQ3:** Does conformal calibration give valid coverage on held-out events?
*   **RQ4:** Does uncertainty-aware allocation reduce unmet demand and regret compared with deterministic allocation?

In answering these questions, the project aims to deliver the following novel contributions to the field of AI for disaster management:

*   **C1:** An uncertainty-aware multimodal flood model utilizing modality dropout for robust performance even when sensor data is missing.
*   **C2:** A methodology for calibrated uncertainty-to-impact propagation utilizing Monte Carlo techniques and conformal prediction.
*   **C3:** A chance-constrained scenario-based allocation algorithm directly driven by the learned predictive distribution, enabling risk-averse decision-making.
*   **C4:** A rigorous evaluation protocol featuring cross-event generalization testing and decision-level metrics (regret, unmet demand) rather than merely predictive metrics (IoU, accuracy).
